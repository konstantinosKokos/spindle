"""
    Contains utility functions to convert the aethel dataset to inputs for the neural proof net.
"""

from __future__ import annotations

import pdb
from dataclasses import dataclass
from .tree import Tree, Unary, Binary, Leaf, Symbol
from collections import defaultdict

MWU = Leaf(Symbol('MWU'))


def index_tree(tree: Tree[Symbol], index: int = 0) -> tuple[Tree[Symbol], int]:
    match tree:
        case Leaf(Symbol('MWU')):
            return tree, index
        case Leaf(Symbol(name, _)):
            return Leaf(Symbol(name, index)), index + 1
        case Unary(symbol, content):
            content, index = index_tree(content, index)
            return Unary(symbol, content), index
        case Binary(symbol, left, right):
            left, index = index_tree(left, index)
            right, index = index_tree(right, index)
            return Binary(symbol, left, right), index
        case _: raise ValueError(f'Unknown tree type: {tree}')


@dataclass
class Sample:
    # todo: distinguish between inference and training samples
    # todo: index-generic atomsets and matrices
    # todo: type-level conversions
    words: list[str]
    trees: list[Tree[Symbol]]
    links: dict[Symbol, Symbol] | None = None
    source: str | None = None
    subset: str | None = None


def extract_unique_symbols(trees: list[Tree[Symbol]]) -> set[tuple[Symbol, int]]:
    return {(symbol.plain(), arity) for tree in trees for symbol, arity in tree.nodes_and_arities()}


def make_symbol_map(symbols: set[tuple[Symbol, int]]) -> tuple[dict[int, Symbol], dict[Symbol, int]]:
    sorted_symbols: list[tuple[Symbol, int | None]] = sorted(symbols, key=lambda s: (s[1], s[0].name))
    id_to_symbol = {i: s for i, s in enumerate([s for s, _ in sorted_symbols])}
    symbol_to_arity = {s: a for s, a in sorted_symbols}
    return id_to_symbol, symbol_to_arity


def write_symbol_map(id_to_symbol: dict[int, Symbol], symbol_to_arity: dict[Symbol, int], path: str):
    with open(path, 'w') as f:
        for i in range(len(id_to_symbol)):
            f.write(f'{i}\t{id_to_symbol[i]}\t{symbol_to_arity[id_to_symbol[i]]}\n')


def whitespace_punct(s: str) -> str:
    return ' '.join(s.replace('\xad', '-').translate(str.maketrans({k: f' {k} ' for k in "!?.,"})).split())


def pad_mwus(words: list[str], types: list[Tree]) -> tuple[list[str], list[Tree]]:
    words = [whitespace_punct(w).split() for w in words]
    types = [[t] + [MWU] * (len(ws) - 1) for ws, t in zip(words, types)]
    return [w for units in words for w in units], [t for units in types for t in units]


def get_word_starts(types: list[Tree]) -> list[int]:
    return [i for i, t in enumerate(types) if t != MWU]


def merge_mwus(words: list[str], types: list[Tree]) -> tuple[list[str], list[Tree]]:
    word_starts = get_word_starts(types)
    ws = [' '.join(words[start:end]) for start, end in zip(word_starts, word_starts[1:] + [len(words) + 1])]
    ts = [types[start] for start in word_starts]
    return ws, ts


def occurrence_count(samples: list[Sample]) -> list[tuple[Tree[Symbol], int]]:
    trees = [tree.fmap(Symbol.plain) for sample in samples for tree in sample.trees[1:]]
    counts = defaultdict(lambda: 0)
    for tree in trees:
        counts[tree] += 1
    return sorted(counts.items(), key=lambda c: c[1], reverse=True)
