import pdb

from dyngraphst.data.tree import Tree, Leaf, Binary, Symbol
from dyngraphst.data.preprocessing import Sample, make_symbol_map, extract_unique_symbols, pad_mwus
import os

from typing import Optional
import re


def string_to_prims(cat: str) -> list[str]:
    primitive_buffer = []
    for i, c in enumerate(cat):
        if c not in {'\\', '/', '(', ')'}:
            primitive_buffer.append(c)
        else:
            if primitive_buffer:
                return [''.join(primitive_buffer), c] + string_to_prims(cat[i + 1:])
            return [c] + string_to_prims(cat[i+1:])
    return [''.join(primitive_buffer)]


def prims_to_tree(prims: list[str]) -> Tree[Symbol]:
    def f(_prims: list[str], history: Optional[Tree[Symbol]]) -> tuple[Tree[Symbol], list[str]]:
        if not _prims:
            return history, []
        if _prims[0] == '(':
            return f(_prims[1:], history)
        if _prims[0] == ')':
            return f(_prims[1:], history)
        if (binary := _prims[0]) in {'\\', '/'}:
            right, rest = f(_prims[1:], history)
            return Binary(Symbol(binary), history, right), rest
        return f(_prims[1:], Leaf(Symbol(_prims[0])))
    return f(prims, None)[0]


def string_to_tree(string: str) -> Tree[Symbol]:
    return prims_to_tree(string_to_prims(string))


def parse_sid(line: str, name: str) -> Sample:
    leaves = re.findall('<L (.*?)>', line)
    cats, _, _, tokens, _ = zip(*[leaf.split() for leaf in leaves])
    trees = [string_to_tree(cat) for cat in cats]
    words, trees = pad_mwus(tokens, trees)
    if top_rule := re.findall('<T (.*?)>', line):
        conclusion = string_to_tree(top_rule[0].split()[0])
    else:
        assert len(trees) == 1
        conclusion = trees[0]
    return Sample(words, [conclusion] + trees, source=name)


def parse_auto(auto: str) -> list[Sample]:
    with open(auto, 'r') as f:
        lines = f.read().split('\n')
    named_parses = [(lines[i], lines[i+1]) for i in range(0, len(lines) - 1, 2)]
    samples = []
    for name, parse in named_parses:
        samples.append(parse_sid(parse, name))
    return samples


def parse_directory(directory: str) -> tuple[list[Sample], list[Sample], list[Sample]]:
    train, dev, test = [], [], []
    for subdir in os.listdir(directory):
        if subdir == '00':
            for file in os.listdir(os.path.join(directory, subdir)):
                if file.endswith('.auto'):
                    dev.extend(parse_auto(os.path.join(directory, subdir, file)))
        elif subdir in {'23'}:
            for file in os.listdir(os.path.join(directory, subdir)):
                if file.endswith('.auto'):
                    test.extend(parse_auto(os.path.join(directory, subdir, file)))
        elif subdir == {'01', '24'}:
            continue
        else:
            for file in os.listdir(os.path.join(directory, subdir)):
                if file.endswith('.auto'):
                    train.extend(parse_auto(os.path.join(directory, subdir, file)))

    with open('./experiments/ccg/atom_map.txt', 'w') as f:
        symbols = extract_unique_symbols([tree for subset in (train, dev, test)
                                          for sample in subset for tree in sample.trees])
        id_to_symbol, symbol_to_arity = make_symbol_map(symbols)
        for i, s in id_to_symbol.items():
            f.write(f'{i}\t{s.name}\t{symbol_to_arity[s]}\n')

    return train, dev, test