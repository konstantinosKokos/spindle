from __future__ import annotations

from dataclasses import dataclass
from .tree import Tree, Leaf, Symbol, Node
from collections import defaultdict

MWU = Leaf(Symbol('MWU'))


@dataclass
class Sample:
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


def pad_mwus(words: list[str], trees: list[Tree[Symbol]]) -> tuple[list[str], list[Tree[Symbol]]]:
    words = [whitespace_punct(w).split() for w in words]
    trees = [[t] + [MWU] * (len(ws) - 1) for ws, t in zip(words, trees)]
    return [w for units in words for w in units], [t for units in trees for t in units]


def get_word_starts(types: list[Tree[Symbol]]) -> list[int]:
    return [i for i, t in enumerate(types) if t != MWU]


def merge_on_word_starts(words: list[str],
                         trees: list[Tree[Node]],
                         word_starts: list[int]) -> tuple[list[str], list[Tree[Node]]]:
    ws = [' '.join(words[start:end]) for start, end in zip(word_starts, word_starts[1:] + [len(words) + 1])]
    ts = [trees[start] for start in word_starts]
    return ws, ts


def occurrence_count(samples: list[Sample]) -> list[tuple[Tree[Symbol], int]]:
    trees = [tree.fmap(Symbol.plain) for sample in samples for tree in sample.trees[1:]]
    counts = defaultdict(lambda: 0)
    for tree in trees:
        counts[tree] += 1
    return sorted(counts.items(), key=lambda c: c[1], reverse=True)


def merge_preds_on_true(preds: list[Tree[Symbol]], indices: list[int]) -> list[Tree[Symbol]]:
    def check_rest(ps: list[Tree[Symbol]]) -> bool:
        return all(p == MWU for p in ps)
    return [preds[i] if check_rest(preds[i+1:end]) else MWU for i, end in zip(indices, indices + [len(preds)])]
