import pdb

from dyngraphst.data.processing import Tree, Symbol, Leaf, Binary, Unary, Sample, pad_mwus

from random import seed, choices

seed(42)

# define atomic categories to easily eval strings to trees
(np, n, s, s_q, s_inf, let, txt, s_ppres, pp_a, pp, s_pass, pp_de, cl_r, s_ppart, s_whq, pp_par, cl_y) = (
    Leaf(Symbol(x)) for x in
    ('np', 'n', 's', 's_q', 's_inf', 'let', 'txt', 's_ppres',
     'pp_a', 'pp', 's_pass', 'pp_de', 'cl_r', 's_ppart', 's_whq', 'pp_par', 'cl_y'))

blank = Leaf(Symbol('blank'))


def dr(mode, left, right) -> Binary[Symbol]:
    return Binary(Symbol(f'/{mode}'), left, right)


def dl(mode, left, right) -> Binary[Symbol]:
    return Binary(Symbol(f'\\{mode}'), left, right)


def dia(mode, content) -> Unary[Symbol]:
    return Unary(Symbol(f'<>{mode}'), content)


def box(mode, content) -> Unary[Symbol]:
    return Unary(Symbol(f'[]{mode}'), content)


def p(mode, left, right) -> Binary[Symbol]:
    return Binary(Symbol(f'(x){mode}'), left, right)


def binarize_structurally(tree: Tree[Symbol]) -> Tree[Symbol]:
    match tree:
        case Leaf(_):
            return tree
        case Binary(outer, left, right):
            return Binary(outer, binarize_structurally(left), binarize_structurally(right))
        case Unary(outer, content):
            return Binary(outer, blank, binarize_structurally(content))
        case _:
            raise ValueError(f'unexpected tree: {tree}')


def binarize_content(tree: Tree[Symbol]) -> Tree[Symbol]:
    match tree:
        case Leaf(_):
            return tree
        case Unary(outer, Unary(inner, content)):
            return Unary(Symbol(f'{outer}◦{inner}'), binarize_content(content))
        case Unary(outer, content):
            return Unary(outer, binarize_content(content))
        case Binary(outer, left, right):
            left = binarize_content(left)
            right = binarize_content(right)
            match left, right:
                case Unary(_, _), Unary(_, _):
                    raise ValueError
                case Unary(inner, content), _:
                    return Binary(Symbol(f'{outer}◦{inner}'), binarize_content(content), right)
                case _, Unary(inner, content):
                    return Binary(Symbol(f'{outer}◦{inner}'), left, binarize_content(content))
                case _, _:
                    return Binary(outer, left, right)
        case _:
            raise ValueError


def parse_line(line: str) -> Sample:
    binarize = binarize_structurally
    tokens = line.split()
    words, _, cats = list(zip(*[t.split('|') for t in tokens]))
    trees_0 = list(map(eval, cats))
    trees = list(map(binarize, trees_0))
    assert not any(isinstance(tree, Unary) for tree in trees)
    words, trees = pad_mwus(words, trees)
    subset = choices(('train', 'dev', 'test'), weights=(0.8, 0.1, 0.1))
    return Sample(words, [txt] + trees, subset=subset[0])


def parse_file(path: str) -> tuple[list[Sample], list[Sample], list[Sample]]:
    with open(path, 'r') as f:
        samples = [parse_line(line) for line in f.readlines()[:-1]]
    return ([sample for sample in samples if sample.subset == 'train'],
            [sample for sample in samples if sample.subset == 'dev'],
            [sample for sample in samples if sample.subset == 'test'])
