from __future__ import annotations

import torch
from typing import Iterator, Callable
from torch import Tensor
from itertools import product, zip_longest, chain, groupby
from ..data.vectorization import TokenizedTrees, TokenizedMatchings, TokenizedMatching, TokenizedSample
from dataclasses import dataclass

from torch.nn.utils.rnn import pad_sequence as _pad_sequence


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


@dataclass
class ParserBatch:
    tensors: list[Tensor]   # each tensor is (batch_size, num_candidates, 2, 3)

    def to(self, device: torch.device) -> ParserBatch:
        return ParserBatch([tensor.to(device) for tensor in self.tensors])


def batchify_parser_inputs(matchings_list: list[TokenizedMatchings],
                           sent_lengths: list[int]) -> ParserBatch:
    def go() -> Iterator[TokenizedMatching]:
        def offset(m: TokenizedMatching) -> TokenizedMatching:
            return [((l_level, l_tree + csum, l_position), (r_level, r_tree + csum, r_position))
                    for (l_level, l_tree, l_position), (r_level, r_tree, r_position) in m]

        csum = 0
        for _, (matchings, sl) in enumerate(zip(matchings_list, sent_lengths)):
            for _, matching in matchings.items():
                yield offset(matching)
            csum += sl
    return ParserBatch([torch.tensor(list(ms))
                        for num_candidates, ms in groupby(sorted(go(), key=len), key=len) if num_candidates > 1])


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
        parser_batch = batchify_parser_inputs(tokenized_matchings, sent_lens)
    else:
        parser_batch = None
    return Batch(encoder_batch, decoder_batch, parser_batch).to(device)


def make_collator(device: torch.device, pad_token_id: int, cls_dist: int) -> Callable[[BatchInput], Batch]:
    def wrapped(batch: BatchInput) -> Batch:
        return collate_fn(batch, device, pad_token_id, cls_dist)
    return wrapped
