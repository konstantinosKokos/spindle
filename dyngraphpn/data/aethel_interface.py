from aethel.frontend import Sample as AethelSample, ProofBank, LexicalPhrase, LexicalItem, Type, Proof
from aethel.mill.nets import (
    FormulaTree, UnaryFT, BinaryFT, LeafFT,
    proof_to_links, links_to_proof, tree_to_type as ft_to_type, AxiomLinks
)

from dyngraphpn.data.tree import Tree, Leaf, Unary, Binary, Symbol
from dyngraphpn.data.processing import Sample, make_symbol_map, extract_unique_symbols, pad_mwus


def ft_to_tree(formula_tree: FormulaTree) -> Tree[Symbol]:
    def convert(ft: FormulaTree) -> Tree[Symbol]:
        match ft:
            case LeafFT(atom, index, _):
                return Leaf(Symbol(atom, index))
            case BinaryFT(left, right, _):
                return Binary(Symbol('->'), convert(left), convert(right))
            case UnaryFT('diamond', UnaryFT('box', content, inner, _), outer, _):
                if inner == outer:
                    return Unary(Symbol('!' + outer), convert(content))
                return Unary(Symbol('◇' + outer), convert(UnaryFT('box', content, inner)))
            case UnaryFT('diamond', content, decoration, _):
                return Unary(Symbol('◇' + decoration), convert(content))
            case UnaryFT('box', content, decoration, _):
                return Unary(Symbol('□' + decoration), convert(content))
            case _:
                raise ValueError

    def merge_adjacent_unaries(tree: Tree[Symbol]) -> Tree[Symbol]:
        match tree:
            case Binary(Symbol(outer), left, right):
                return Binary(Symbol(outer), merge_adjacent_unaries(left), merge_adjacent_unaries(right))
            case Unary(Symbol(outer), Unary(Symbol(inner), nested)):
                if outer.startswith('!'):
                    return Unary(Symbol(outer + inner), merge_adjacent_unaries(nested))
                return Unary(Symbol(outer), merge_adjacent_unaries(Unary(Symbol(inner), nested)))
            case Unary(Symbol(outer), content):
                return Unary(Symbol(outer), merge_adjacent_unaries(content))
            case _:
                return tree

    def binarize(tree: Tree[Symbol]) -> Tree[Symbol]:
        match tree:
            case Leaf(_):
                return tree
            case Unary(Symbol(outer), Unary(Symbol(inner), nested)):
                return Unary(Symbol(outer), binarize(Unary(Symbol(inner), nested)))
            case Unary(Symbol(outer), Binary(Symbol(inner), left, right)):
                return Binary(Symbol(outer + inner), binarize(left), binarize(right))
            case Binary(Symbol(outer), Unary(Symbol(inner), left), right):
                return Binary(Symbol(outer + inner), binarize(left), binarize(right))
            case Binary(Symbol(outer), left, right):
                return Binary(Symbol(outer), binarize(left), binarize(right))
            case _:
                raise ValueError(f'Unknown tree type: {tree}')

    return binarize(merge_adjacent_unaries(convert(formula_tree)))


def tree_to_ft(tree: Tree[Symbol], polarity: bool) -> FormulaTree:
    def debinarize(t: Tree[Symbol]) -> Tree[Symbol]:
        def nested_unary(_tree: Tree[Symbol], modalities: list[str]) -> Tree[Symbol]:
            match modalities:
                case []:
                    return _tree
                case [m, *ms]:
                    return Unary(Symbol(m), nested_unary(_tree, ms))

        def split_modality(compound: str) -> list[str]:
            def go(pref: str, chars: str) -> list[str]:
                if len(chars):
                    init, cs = chars[0], chars[1:]
                    if init in {'!', '◇', '□'}:
                        return [pref, *go(init, cs)]
                    return go(pref + init, cs)
                return [pref]

            return go(compound[0], compound[1:])

        match t:
            case Binary(Symbol(sym), left, right):
                if sym == '->':
                    return Binary(Symbol('->'), debinarize(left), debinarize(right))
                if sym.startswith('->'):
                    modality = sym.lstrip('->')
                    return Binary(Symbol('->'), nested_unary(debinarize(left), split_modality(modality)),
                                  debinarize(right))
                if sym.endswith('->'):
                    modality = sym.rstrip('->')
                    return nested_unary(Binary(Symbol('->'), debinarize(left), debinarize(right)),
                                        split_modality(modality))
                raise ValueError(f'Cannot parse binary symbol: {sym}')
            case _:
                return t

    def convert(t: Tree[Symbol], p: bool) -> FormulaTree:
        match t:
            case Leaf(symbol):
                return LeafFT(symbol.name, symbol.index, p)
            case Unary(Symbol(s), content):
                match tuple(s):
                    case ('◇', *ds):
                        return UnaryFT('diamond', convert(content, p), ''.join(ds))
                    case ('□', *ds):
                        return UnaryFT('box', convert(content, p), ''.join(ds))
                    case ('!', *ds):
                        return UnaryFT('diamond',
                                       UnaryFT('box', convert(content, p), ''.join(ds)),
                                       ''.join(ds))
                    case _:
                        raise ValueError
            case Binary(_, left, right):
                return BinaryFT(convert(left, not p), convert(right, p))
            case _:
                raise ValueError(f'Unknown tree type: {t}')

    return convert(debinarize(tree), polarity)


def tree_to_type(tree: Tree) -> Type:
    return ft_to_type(tree_to_ft(tree, True))


def from_aethel(sample: AethelSample) -> Sample:
    def leaf_to_symbol(leaf: LeafFT) -> Symbol:
        return Symbol(leaf.atom, leaf.index)

    links, participating_trees, conclusion = proof_to_links(sample.proof)
    trees = [ft_to_tree(participating_trees[i])
             if i in participating_trees.keys()
             else Leaf(Symbol(sample.lexical_phrases[i].type.sign))  # type: ignore
             for i in range(len(sample.lexical_phrases))]
    words = [lp.string for lp in sample.lexical_phrases]
    words, trees = pad_mwus(words, trees)
    trees = [ft_to_tree(conclusion)] + trees
    links = {leaf_to_symbol(neg): leaf_to_symbol(pos) for neg, pos in links.items()}
    return Sample(
        words=words,
        trees=trees,
        links=links,
        source=sample.name,
        subset=sample.subset)


def preprocess(load_path: str = '../lassy-tlg-extraction/data/aethel_1.0.0a5.pickle') \
        -> tuple[list[Sample], list[Sample], list[Sample]]:
    aethel_samples = ProofBank.load_data(load_path)
    processed_samples = []
    for aethel_sample in aethel_samples:
        try:
            processed_samples.append(from_aethel(aethel_sample))
        except ValueError:
            continue
    print(f'Converted {len(processed_samples)} of {len(aethel_samples)}')
    with open('./data/atom_map.tsv', 'w') as f:
        symbols = extract_unique_symbols([tree for s in processed_samples for tree in s.trees])
        id_to_symbol, symbol_to_arity = make_symbol_map(symbols)
        for i, s in id_to_symbol.items():
            f.write(f'{i}\t{s.name}\t{symbol_to_arity[s]}\n')
    return ([d for d in processed_samples if d.subset == 'train'],
            [d for d in processed_samples if d.subset == 'dev'],
            [d for d in processed_samples if d.subset == 'test'])
