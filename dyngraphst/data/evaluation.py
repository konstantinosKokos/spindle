import pdb
import pickle
from operator import eq

from .vectorization import AtomTokenizer, Tree, Symbol, Leaf, Binary
from .processing import MWU, get_word_starts

from collections import defaultdict


def levels_to_trees(decoder_output: list[list[int]], atoken: AtomTokenizer) -> list[Tree[Symbol]]:
    levels = [[atoken.id_to_token[i] for i in level] for level in decoder_output]
    fringe: list[Tree[Symbol]] = [Leaf(symbol) for symbol in levels[-1]]
    for level in reversed(levels[:-1]):
        stack = list(reversed(fringe))
        fringe = []
        for symbol in level:
            if atoken.symbol_arities[symbol] == 2:
                left = stack.pop()
                right = stack.pop()
                fringe.append(Binary(symbol, left, right))
            else:
                fringe.append(Leaf(symbol))
    return fringe


def trees_to_frames(trees: list[Tree[Symbol]], splitpoints: list[int]) -> list[list[Tree[Symbol]]]:
    splitpoints = [1 + s for s in splitpoints]
    return [trees[start:end] for start, end in zip([0] + splitpoints, splitpoints)]


def merge_preds_on_true(preds: list[Tree[Symbol]], indices: list[int]) -> list[Tree[Symbol]]:
    return [preds[i] if all([p == MWU for p in preds[i+1:end]]) else MWU  # force wrong on bad MWU
            for i, end in zip(indices, indices + [len(preds)])]


def evaluate_results_file(results_file: str, atom_map_path: str, occurrence_file: str | None = None):
    with open(results_file, 'rb') as f:
        outs = pickle.load(f)

    atom_tokenizer = AtomTokenizer.from_file(atom_map_path)

    pred_frames = [frame for out in outs for frame in trees_to_frames(levels_to_trees(out[0], atom_tokenizer), out[2])]
    gold_frames = [frame for out in outs for frame in trees_to_frames(levels_to_trees(out[1], atom_tokenizer), out[2])]
    mwp_indices = [get_word_starts(frame) for frame in gold_frames]
    gold_frames = [merge_preds_on_true(frame, mwp_indices[i])[1:] for i, frame in enumerate(gold_frames)]
    pred_frames = [merge_preds_on_true(frame, mwp_indices[i])[1:] for i, frame in enumerate(pred_frames)]

    pred_trees = sum(pred_frames, [])
    gold_trees = sum(gold_frames, [])

    assert MWU not in gold_trees

    # token-wise
    print('Token-wise accuracy:')
    s = sum(map(eq, pred_trees, gold_trees))
    print(f'{s / len(gold_trees)} ({s} / {len(gold_trees)}) ({len(set(gold_trees))})')
    # frame-wise
    print('Frame-wise accuracy:')
    print(f'{sum(map(eq, pred_frames, gold_frames)) / len(gold_frames)} ({len(gold_frames)})')

    if occurrence_file is not None:
        with open(occurrence_file, 'rb') as f:
            nzero = pickle.load(f)
            occurrence_counts = defaultdict(lambda: 0, nzero)

    # occurrence-wise
    rest = [i for i in range(len(pred_trees)) if occurrence_counts[gold_trees[i]] >= 100]
    mrare = [i for i in range(len(pred_trees)) if 10 <= occurrence_counts[gold_trees[i]] < 100]
    rare = [i for i in range(len(pred_trees)) if 0 < occurrence_counts[gold_trees[i]] < 10]
    unk = [i for i in range(len(pred_trees)) if occurrence_counts[gold_trees[i]] == 0]

    print('100+ accuracy:')
    print(f'{sum(map(eq, [pred_trees[i] for i in rest], [gold_trees[i] for i in rest])) / len(rest)} ({len(rest)})')
    print('10-99 accuracy:')
    s = sum(map(eq, [pred_trees[i] for i in mrare], gs := [gold_trees[i] for i in mrare]))
    print(f'{s / len(mrare)} ({s}/{len(mrare)}) {len(set(gs))}')
    print('0-9 accuracy:')
    s = sum(map(eq, [pred_trees[i] for i in rare], gs := [gold_trees[i] for i in rare]))
    print(f'{s / len(rare)} ({s}/{len(rare)}) {len(set(gs))}')
    print('unknown accuracy:')
    s = sum(map(eq, [pred_trees[i] for i in unk], gs := [gold_trees[i] for i in unk]))
    print(f'{s / len(unk)} ({s}/{len(unk)}) {len(set(gs))}')
