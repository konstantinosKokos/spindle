from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from torch.utils.data import DataLoader

from typing import Iterator, Callable
from dataclasses import dataclass
from itertools import product, zip_longest, chain, groupby

from ..data.tokenization import (Tree, Binary, Leaf, Symbol,
                                 TokenizedTrees, TokenizedMatchings, TokenizedSample, TokenizedSamples)


def pad_sequence(sequence: list[Tensor], padding_value: int = 0) -> Tensor:
    return _pad_sequence(sequence, batch_first=True, padding_value=padding_value)


def sent_lengths_to_edges(sent_lengths: list[int], cls_distance: int) -> Iterator[tuple[tuple[int, int], int]]:
    csum = 0
    for sl in sent_lengths:
        for i, j in product(range(sl), range(sl)):
            yield (i + csum, j + csum), i - j if i != 0 and j != 0 else cls_distance
        csum += sl


@dataclass
class EncoderBatch:
    token_ids:      Tensor
    atn_mask:       Tensor
    cluster_ids:    Tensor
    edge_index:     Tensor
    edge_attr:      Tensor

    def to(self, device: torch.device) -> EncoderBatch:
        return EncoderBatch(self.token_ids.to(device),
                            self.atn_mask.to(device),
                            self.cluster_ids.to(device),
                            self.edge_index.to(device),
                            self.edge_attr.to(device))


def batchify_encoder_inputs(token_ids: list[list[int]],
                            token_clusters: list[list[int]],
                            pad_token_id: int,
                            cls_dist: int = -999) -> tuple[EncoderBatch, list[int]]:
    sent_lens = [max(t) + 1 for t in token_clusters]
    edge_index, edge_distance = zip(*sent_lengths_to_edges(sent_lens, cls_dist))
    token_ids = pad_sequence(list(map(torch.tensor, token_ids)), padding_value=pad_token_id)
    token_clusters = pad_sequence(list(map(torch.tensor, token_clusters)), padding_value=-1)
    offsets = (token_clusters.max(dim=-1).values + 1).cumsum(dim=-1).roll(1, 0).unsqueeze(-1)
    offsets[0] = 0
    return (EncoderBatch(token_ids,
                         token_ids.ne(pad_token_id).long(),
                         token_clusters + offsets,
                         torch.tensor(edge_index).t(),
                         torch.tensor(edge_distance)),
            sent_lens)


@dataclass
class DecoderBatch:
    edge_index: list[Tensor]
    token_ids:  list[Tensor]
    pos_ids:    list[Tensor]

    def to(self, device: torch.device) -> DecoderBatch:
        return DecoderBatch([e.to(device) for e in self.edge_index],
                            [t.to(device) for t in self.token_ids],
                            [p.to(device) for p in self.pos_ids])


def batchify_decoder_inputs(trees: list[list[TokenizedTrees]]) -> DecoderBatch:
    def go() -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        depths = zip_longest(*chain.from_iterable(trees), fillvalue=[])
        for depth in depths:
            root_ids, token_ids, positions = zip(*((root_id, token_id, position)
                                                   for root_id, nodes in enumerate(depth)
                                                   for token_id, position, _, _ in nodes))
            yield (torch.vstack([torch.tensor(root_ids), torch.arange(len(root_ids))]),
                   torch.tensor(token_ids),
                   torch.tensor(positions))

    edge_index, token_ids, positions = zip(*go())
    return DecoderBatch(edge_index, token_ids, positions)


# row, atom_id, negative_link_indices, positive_link_indices
BackPointer = list[tuple[int, str | int, list[int], list[int]]]


@dataclass
class ParserBatch:
    indices: list[Tensor]            # each element is a (b, n, 2, 2) tensor
    backpointers: list[BackPointer]  # a bp for each indexing tensor (mostly useless for training)

    def to(self, device: torch.device) -> ParserBatch:
        return ParserBatch([tensor.to(device) for tensor in self.indices], self.backpointers)


def batchify_parser_inputs(matchings_list: list[TokenizedMatchings],
                           decoder_batch: DecoderBatch,
                           sent_lens: list[int]) -> ParserBatch:
    def go() -> Iterator[tuple[tuple[int, str | int, list[int], list[int]],
                               list[tuple[tuple[int, int], tuple[int, int]]]]]:
        def batchify_token_pos(_: int, level: int, root: int, pos_id: int) -> tuple[int, int]:
            level_filter = decoder_batch.edge_index[level][0] == root + csum
            pos_filter = decoder_batch.pos_ids[level] == pos_id
            index = (level_filter & pos_filter).nonzero()[0].item()
            return level, index

        def gather_link_ids(token_pos: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]) \
                -> tuple[list[int], list[int]]:
            negs, pos = zip(*token_pos)
            return [link_idx for link_idx, _, _, _, in negs], [link_idx for link_idx, _, _, _, in pos]

        csum = 0
        for s_id, (matchings, sent_len) in enumerate(zip(matchings_list, sent_lens)):
            for atom_id, matching in matchings.items():
                b_pos = [(batchify_token_pos(*neg), batchify_token_pos(*pos)) for neg, pos in matching]
                yield (s_id, atom_id, *gather_link_ids(matching)), b_pos
            csum += sent_len

    arranged = groupby(sorted(go(), key=lambda x: len(x[-1])), key=lambda x: len(x[-1]))
    backpointers, indices = zip(*[tuple(zip(*ms)) for nc, ms in arranged if nc > 1])
    return ParserBatch([torch.tensor(group) for group in indices],
                       backpointers)


_Matrix = list[tuple[str, tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]]]


def ptrees_to_candidates(ptrees: list[Tree[tuple[Symbol, tuple[int, int]]]]) -> ParserBatch | None:
    def go_tree(_ptree: Tree[tuple[Symbol, tuple[int, int]]],
                polarity: bool) -> Iterator[tuple[str, int, int, int | None, bool]]:
        match _ptree:
            case Binary(_, left, right):
                yield from go_tree(left, not polarity)
                yield from go_tree(right, polarity)
            case Leaf((sym, (level_idx, n_idx))):
                yield sym.name, sym.index, level_idx, n_idx, polarity
            case _:
                raise ValueError

    def go_seq() -> Iterator[tuple[str, int, int, int | None, bool]]:
        (conclusion, *assignments) = ptrees
        for ptree in assignments:
            yield from go_tree(ptree, True)
        yield from go_tree(conclusion, False)

    def nc(t: tuple[str, tuple[list[tuple, tuple]]]) -> int:
        _, (ns, _) = t
        return len(ns)

    def split_by_polarity(*items: tuple[str, int, int, int, bool]) \
            -> tuple[list[tuple[str, int, int, int]], list[tuple[str, int, int, int]]]:
        return [item[1:4] for item in items if not item[4]], [item[1:4] for item in items if item[4]]

    def matrices_to_indices_and_bps(ms: list[_Matrix]) -> tuple[Tensor, BackPointer]:
        def matrix_to_tensor(ns: list[tuple[int, int, int]], ps: list[tuple[int, int, int]]) -> Tensor:
            pairs = list(zip(ns, ps))
            return torch.tensor(pairs)[..., 1:]

        def matrix_to_bp(ns: list[tuple[int, int, int]], ps: list[tuple[int, int, int]]):
            return [n[0] for n in ns], [p[0] for p in ps]

        names, rest = zip(*ms)
        tensor = torch.stack([matrix_to_tensor(*x) for x in rest])
        link_indices = [matrix_to_bp(*x) for x in rest]
        bp = [(0, name, nlinks, plinks) for name, (nlinks, plinks) in zip(names, link_indices)]
        return tensor, bp

    # ignore leaves with no symbol index
    seq = filter(lambda x: x[1] is not None, go_seq())
    grouped = {atom: split_by_polarity(*group)
               for atom, group in groupby(sorted(seq, key=lambda item: item[0]), key=lambda item: item[0])}
    if all(len(neg) == len(pos) for neg, pos in grouped.values()):
        matrix_groups: list[list[tuple[str, tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]]]]
        matrix_groups = [list(gs) for _, gs in groupby(sorted(grouped.items(), key=nc), key=nc)]

        indices: list[Tensor] = []
        backpointers: list[BackPointer] = []
        matrices: list[_Matrix]
        for matrices in matrix_groups:
            indexing_tensor, backpointer = matrices_to_indices_and_bps(matrices)
            indices.append(indexing_tensor)
            backpointers.append(backpointer)
        return ParserBatch(indices, backpointers)
    return None


BatchInput = list[TokenizedSample]


@dataclass
class Batch:
    encoder_batch: EncoderBatch
    decoder_batch: DecoderBatch
    parser_batch:  ParserBatch | None

    def to(self, device: torch.device) -> Batch:
        return Batch(self.encoder_batch.to(device),
                     self.decoder_batch.to(device),
                     self.parser_batch.to(device) if self.parser_batch is not None else None)


def collate_fn(batch_items: BatchInput,
               device: torch.device,
               pad_token_id: int,
               cls_dist: int = -999) -> Batch:
    tokenized_sentences, tokenized_trees, tokenized_matchings = zip(*batch_items)
    sentential_token_ids, sentential_cluster_ids = zip(*tokenized_sentences)
    encoder_batch, sent_lens = batchify_encoder_inputs(sentential_token_ids,
                                                       sentential_cluster_ids,
                                                       pad_token_id,
                                                       cls_dist)
    decoder_batch = batchify_decoder_inputs(tokenized_trees)
    if tokenized_matchings is not None:
        parser_batch = batchify_parser_inputs(tokenized_matchings, decoder_batch, sent_lens)
    else:
        parser_batch = None
    return Batch(encoder_batch, decoder_batch, parser_batch).to(device)


def make_collator(device: torch.device, pad_token_id: int, cls_dist: int) -> Callable[[BatchInput], Batch]:
    def wrapped(batch: BatchInput) -> Batch:
        return collate_fn(batch, device, pad_token_id, cls_dist)
    return wrapped


def make_loader(data: TokenizedSamples,
                device: torch.device,
                pad_token_id: int,
                max_seq_len: int,
                batch_size: int = 16,
                sort: bool = False,
                cls_dist: int = -999) -> DataLoader:
    subset = [sample for sample in data if len(sample[0][0]) <= max_seq_len]
    cfn = make_collator(device, pad_token_id=pad_token_id, cls_dist=cls_dist)
    if sort:
        subset = sorted(subset, key=lambda sample: len(sample[0][0]))
    return DataLoader(subset, shuffle=not sort, collate_fn=cfn, batch_size=batch_size)
