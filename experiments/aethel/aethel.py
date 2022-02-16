from LassyExtraction.extraction import Atoms
from LassyExtraction.aethel import Sample as AethelSample, aethel
from LassyExtraction.mill.nets import (Tree as FTree, Unary as FUnary, Binary as FBinary, Leaf as FLeaf,
                                       term_to_links, type_to_tree)


from dyngraphst.data.tree import Tree, Leaf, Unary, Binary, Symbol
from dyngraphst.data.preprocessing import Sample, make_symbol_map, extract_unique_symbols, pad_mwus


def formula_tree_to_tree(formula_tree: FTree) -> Tree[Symbol]:
    match formula_tree:
        case FLeaf(atom, _, index):
            return Leaf(Symbol(atom, index))
        case FUnary(_, modality, decoration, content):
            return Unary(Symbol(modality + decoration), formula_tree_to_tree(content))
        case FBinary(_, left, right):
            return Binary(Symbol('->'), formula_tree_to_tree(left), formula_tree_to_tree(right))
        case _: raise ValueError(f'Unknown formula tree type: {formula_tree}')


def tree_to_formula_tree(tree: Tree[Symbol], polarity: bool = True) -> FTree:
    match tree:
        case Leaf(symbol):
            return FLeaf(symbol.name, polarity, symbol.index)
        case Unary(symbol, content):
            return FUnary(polarity, symbol.name[0], symbol.name[1:], tree_to_formula_tree(content, polarity))
        case Binary(_, left, right):
            return FBinary(polarity, tree_to_formula_tree(left, not polarity), tree_to_formula_tree(right, polarity))
        case _: raise ValueError(f'Unknown tree type: {tree}')


def binarize(tree: Tree[Symbol]) -> Tree[Symbol]:
    match tree:
        case Leaf(_):
            return tree
        case Unary(Symbol(outer), Binary(Symbol(inner), left, right)):
            return Binary(Symbol(outer + inner), binarize(left), binarize(right))
        case Binary(Symbol(outer), Unary(Symbol(inner), left), right):
            return Binary(Symbol(outer + inner), binarize(left), binarize(right))
        case Binary(symbol, left, right):
            return Binary(symbol, binarize(left), binarize(right))
        case _: raise ValueError(f'Unknown tree type: {tree}')


def debinarize(tree: Tree[Symbol]) -> Tree[Symbol]:
    match tree:
        case Leaf(_):
            return tree
        case Binary(Symbol(outer), left, right):
            if outer == '->':
                return Binary(Symbol(outer), debinarize(left), debinarize(right))
            if outer.startswith('->'):
                return Binary(Symbol('->'), Unary(Symbol(outer[2:]), debinarize(left)), debinarize(right))
            elif outer.endswith('->'):
                return Unary(Symbol(outer[:-2]), Binary(Symbol('->'), debinarize(left), debinarize(right)))
            else:
                raise ValueError(f'Unknown binary symbol: {outer}')
        case _: raise ValueError(f'Unknown tree type: {tree}')


def from_aethel(sample: AethelSample) -> Sample:
    premises = sample.premises
    links, participating_trees = term_to_links(sample.proof)
    trees = [formula_tree_to_tree(participating_trees[i])
             if i in participating_trees.keys()
             else Leaf(Symbol(premises[i].type.sign))  # type: ignore
             for i in range(len(sample.premises))]
    words = [p.word for p in premises]
    words, trees = pad_mwus(words, trees)
    trees = [formula_tree_to_tree(type_to_tree(type(sample.proof), False)[0])] + trees
    return Sample(
        words=words,
        trees=[binarize(tree) for tree in trees],
        links={formula_tree_to_tree(neg): formula_tree_to_tree(pos) for neg, pos in links.items()},
        source=sample.name,
        subset=sample.subset)


def to_aethel(sample: Sample) -> tuple[dict[int, int], list[FTree]]:
    # todo: mwu-unification, punct ignoring
    links = {tree_to_formula_tree(neg, False): tree_to_formula_tree(pos) for neg, pos in sample.links.items()}
    trees = [tree_to_formula_tree(debinarize(t)) for t in sample.trees]
    return links, trees


def preprocess(load_path: str = '../lassy-tlg-extraction/data/aethel.pickle') \
        -> tuple[list[Sample], list[Sample], list[Sample]]:
    data = [from_aethel(d) for d in aethel.load_data(load_path)]
    with open('./atom_map.txt', 'w') as f:
        symbols = extract_unique_symbols([tree for s in data for tree in s.trees])
        id_to_symbol, symbol_to_arity = make_symbol_map(symbols)
        for i, s in id_to_symbol.items():
            f.write(f'{i}\t{s.name}\t{symbol_to_arity[s]}\n')
    return ([d for d in data if d.subset == 'train'],
            [d for d in data if d.subset == 'dev'],
            [d for d in data if d.subset == 'test'])
