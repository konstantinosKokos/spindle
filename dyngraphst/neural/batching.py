import pdb

import torch
from typing import Iterator, Callable
from torch import Tensor
from itertools import product, zip_longest

from torch.nn.utils.rnn import pad_sequence as _pad_sequence


def pad_sequence(sequence: list[Tensor], **kwargs) -> Tensor:
    return _pad_sequence(sequence, batch_first=True, **kwargs)


def sent_lengths_to_edges(sent_lengths: list[int], cls_distance: int) -> Iterator[tuple[tuple[int, int], int]]:
    csum = 0
    for sl in sent_lengths:
        for i, j in product(range(sl), range(sl)):
            yield (i + csum, j + csum), i - j if i != 0 and j != 0 else cls_distance
        csum += sl


DecoderItem = list[list[list[list[tuple[int, int, int]]]]]


def batchify_decoder_inputs(trees: list[DecoderItem]) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]:
    concatenated = sum(trees, [])
    depths = zip_longest(*[t for t in concatenated], fillvalue=[])
    for depth in depths:
        root_ids, symbols, positions, numel = zip(*((root_id, symbol, position, numel)
                                                    for root_id, nodes in enumerate(depth)
                                                    for symbol, position, numel in nodes))
        yield (torch.vstack([torch.tensor(root_ids), torch.arange(len(root_ids))]),
               torch.tensor(symbols),
               torch.tensor(positions),
               torch.tensor(numel))


def batchify_encoder_inputs(token_ids: list[Tensor],
                            token_clusters: list[Tensor],
                            pad_token_id: int,
                            cls_dist: int = -999) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    edge_index, edge_distance = list(zip(*sent_lengths_to_edges([max(t) + 1 for t in token_clusters], cls_dist)))
    token_ids = pad_sequence([t for t in token_ids], padding_value=pad_token_id)
    token_clusters = pad_sequence([t for t in token_clusters], padding_value=-1)
    offsets = (token_clusters.max(dim=-1).values + 1).cumsum(dim=-1).roll(1, 0).unsqueeze(-1)
    offsets[0] = 0
    return (
        token_ids,
        token_ids.ne(pad_token_id).long(),
        token_clusters + offsets,
        torch.tensor(edge_index).t(),
        torch.tensor(edge_distance))


EncoderItem = tuple[list[Tensor], list[Tensor]]
BatchItem = tuple[EncoderItem, DecoderItem]
BatchItems = list[BatchItem]
Batch = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[Tensor], list[Tensor], list[Tensor], list[Tensor]]


def make_collator(device: str, pad_token_id: int, cls_dist: int) -> Callable[[BatchItems], Batch]:
    def wrapped(batch: BatchItems) -> Batch:
        return collate_fn(batch, device, pad_token_id, cls_dist)
    return wrapped


def collate_fn(batch_items: BatchItems,
               device: str,
               pad_token_id: int,
               cls_dist: int = -999) -> Batch:
    encoder_inputs, decoder_inputs = zip(*batch_items)
    token_ids, token_clusters = zip(*encoder_inputs)

    token_ids, atn_mask, token_clusters, root_edge_index, root_edge_dist = \
        batchify_encoder_inputs(token_ids, token_clusters, pad_token_id, cls_dist)
    root_to_node_index, node_ids, node_pos, numels = list(zip(*batchify_decoder_inputs(decoder_inputs)))
    return (token_ids.to(device),
            atn_mask.to(device),
            token_clusters.to(device),
            root_edge_index.to(device),
            root_edge_dist.to(device),
            [rtn.to(device) for rtn in root_to_node_index],
            [nid.to(device) for nid in node_ids],
            [np.to(device) for np in node_pos],
            [n.to(device) for n in numels])
