import pdb

import torch

from dyngraphpn.data.tokenization import AtomTokenizer, load_data, group_trees
from dyngraphpn.data.processing import merge_preds_on_true, get_word_starts
from dyngraphpn.neural.batching import make_loaders
from dyngraphpn.neural.model import Parser

from operator import eq

from transformers import BertConfig

from time import time

MAX_DIST = 6
MAX_SEQ_LEN = 199
NUM_SYMBOLS = 80
FIRST_BINARY = 31
MAX_DEPTH = 15
BATCH_SIZE = 64


def evaluate(model_path: str,
             device: torch.device = 'cuda',
             encoder_config_path: str = './data/bert_config.json',
             atom_map_path: str = './data/atom_map.tsv',
             bert_type: str = 'bert',
             data_path: str = './data/vectorized.p',
             num_classes: int = NUM_SYMBOLS,
             max_dist: int = MAX_DIST,
             max_seq_len: int = MAX_SEQ_LEN,
             pad_token_id: int = 3,
             max_depth: int = MAX_DEPTH,
             sep_token_id: int = 2,
             first_binary: int = FIRST_BINARY,
             test_set: bool = False,
             batch_size: int = BATCH_SIZE):
    model = Parser(num_classes=num_classes,
                   max_dist=max_dist,
                   encoder_config_or_name=BertConfig.from_json_file(encoder_config_path),  # type: ignore
                   bert_type=bert_type,
                   sep_token_id=sep_token_id).to(device)

    model.load(model_path, map_location=device)
    model.eval()
    data = load_data(data_path)
    dls = make_loaders(data, device, pad_token_id=pad_token_id, max_seq_len=max_seq_len, batch_size_dev=batch_size)
    dl = dls[2 if test_set else 1]
    model.path_encoder.precompute(2 ** max_depth + 1)
    atokenizer = AtomTokenizer.from_file(atom_map_path)
    start = time()
    with torch.no_grad():
        pred_frames = []
        gold_frames = []

        for batch in dl:
            (preds, decoder_reprs, node_pos) \
                = model.forward_dev(input_ids=batch.encoder_batch.token_ids,
                                    attention_mask=batch.encoder_batch.atn_mask,
                                    token_clusters=batch.encoder_batch.cluster_ids,
                                    root_edge_index=batch.encoder_batch.edge_index,
                                    root_dist=batch.encoder_batch.edge_attr,
                                    max_type_depth=max_depth,
                                    first_binary=first_binary)
            splitpoints = (1 + batch.encoder_batch.cluster_ids.max(dim=1).values).tolist()
            pred_frames.extend(frame for frame in
                               group_trees(atokenizer.levels_to_trees([o.tolist() for o in preds]),
                                           splitpoints))
            gold_frames.extend(frame for frame in
                               group_trees(atokenizer.levels_to_trees([n.tolist() for n in batch.decoder_batch.token_ids]),
                                           splitpoints))
    end = time()
    print(f'Decoding took {end - start} seconds.')
    mwp_indices = [get_word_starts(frame) for frame in gold_frames]
    gold_frames = [merge_preds_on_true(frame, word_starts) for frame, word_starts in zip(gold_frames, mwp_indices)]
    pred_frames = [merge_preds_on_true(frame, word_starts) for frame, word_starts in zip(pred_frames, mwp_indices)]
    pred_trees = [p for frame in pred_frames for p in frame]
    gold_trees = [p for frame in gold_frames for p in frame]
    # token-wise
    print('Token-wise accuracy:')
    s = sum(map(eq, pred_trees, gold_trees))
    print(f'{s / len(gold_trees)} ({s} / {len(gold_trees)}) ({len(set(gold_trees))})')
    # frame-wise
    print('Frame-wise accuracy:')
    print(f'{sum(map(eq, pred_frames, gold_frames)) / len(gold_frames)} ({len(gold_frames)})')
