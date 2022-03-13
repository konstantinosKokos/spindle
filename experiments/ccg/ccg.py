import pdb

from dyngraphst.data.tree import Tree, Leaf, Binary, Symbol
from dyngraphst.data.processing import Sample, make_symbol_map, extract_unique_symbols, pad_mwus
import os

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
    return [''.join(primitive_buffer)] if primitive_buffer else []


def parenthesize(prims: list[str]) -> list[str]:
    return prims if len(prims) == 1 else ['('] + prims + [')']


def prims_to_tree(prims: list[str]) -> Tree[Symbol]:
    if len(prims) == 1:
        return Leaf(Symbol(prims[0]))
    prims = parenthesize(prims)
    stack = []
    while prims:
        prim = prims.pop(0)
        if prim == '(':
            pass
        elif prim == ')':
            right = stack.pop()
            op = stack.pop()
            left = stack.pop()
            stack.append(Binary(op, left, right))
        elif prim in {'\\', '/'}:
            stack.append(Symbol(prim))
        else:
            stack.append(Leaf(Symbol(prim)))
    assert len(stack) == 1
    return stack[0]


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
        elif subdir in {'01', '22', '24'}:
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


def parse_prop(prop: str) -> Sample:
    with open(prop, 'r') as f:
        lines = iter(f.read().split('\n'))
    words, trees, line = [], [], next(lines)
    while not line.startswith('ccg'):
        line = next(lines)
    line = next(lines)
    conclusion = string_to_tree(line.split('(')[1].rstrip(','))
    for line in lines:
        if line.lstrip().startswith('t(,'):
            words.append(',')
            trees.append(Leaf(Symbol(',')))
        elif line.lstrip().startswith('t('):
            rest = line.split('t(')[1].split(', ')
            trees.append(string_to_tree(rest[0]))
            words.append(rest[1][1:-1].rstrip())
    words, trees = pad_mwus(words, trees)
    trees = [conclusion] + trees

    def f(symbol: Symbol) -> Symbol:
        name = symbol.name
        if ':' in name and name != ':':
            l, r = name.split(':')
            return Symbol(f'{l.upper()}[{r}]' if r != 'X' else 'S')
        return Symbol('conj' if name == 'conj' else 'N' if name == 'np, n' else name.upper())

    trees = [tree.fmap(f) for tree in trees]

    return Sample(words, trees, source=prop)


def parse_pmb_dir(directory: str) -> list[Sample]:
    samples = []
    for file in os.listdir(directory):
        for subdir in os.listdir(os.path.join(directory, file)):
            for prop in os.listdir(os.path.join(directory, file, subdir)):
                if prop.endswith('.tags'):
                    samples.append(parse_prop(os.path.join(directory, file, subdir, prop)))
    return samples
