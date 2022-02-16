import pdb
import pickle
from operator import eq

from ..data.vectorization import AtomTokenizer, Tree, Symbol, Leaf, Binary

atoken = AtomTokenizer.from_file()


def levels_to_trees(decoder_output: list[list[int]]) -> list[Tree[Symbol]]:
    levels = [[atoken.id_to_token[i] for i in level] for level in decoder_output]
    fringe: list[Tree[Symbol]] = [Leaf(symbol) for symbol in levels[-1]]
    for level in reversed(levels[:-1]):
        stack = list(reversed(fringe))
        fringe = []
        for symbol in level:
            if atoken.symbol_arities[symbol] == 2:
                right = stack.pop()
                left = stack.pop()
                fringe.append(Binary(symbol, left, right))
            else:
                fringe.append(Leaf(symbol))
    return fringe


def trees_to_frames(trees: list[Tree[Symbol]], splitpoints: list[int]) -> list[list[Tree[Symbol]]]:
    return [trees[start:end] for start, end in zip([0] + splitpoints, splitpoints)]


def evaluate_results_file(results_file: str):
    with open(results_file, 'rb') as f:
        outs = pickle.load(f)
    preds = [levels_to_trees(out[0]) for out in outs]
    truths = [levels_to_trees(out[1]) for out in outs]
    pred_frames = [trees_to_frames(preds[i], outs[i][2]) for i in range(len(outs))]
    true_frames = [trees_to_frames(truths[i], outs[i][2]) for i in range(len(outs))]
    for p, t in zip(sum(preds, []), sum(truths, [])):
        if p != t:
            print(f'{p} |||  {t}')
    print(sum(map(eq, sum(preds, []), sum(truths, [])))/len(sum(truths, [])))
    print(sum(map(eq, sum(pred_frames, []), sum(true_frames, [])))/len(sum(pred_frames, [])))