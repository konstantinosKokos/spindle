from __future__ import annotations
from .processing import Sample, Symbol, whitespace_punct
from .tree import Tree, Leaf, Unary, Binary, Node

from itertools import groupby
from transformers import BertTokenizer, RobertaTokenizer, CamembertTokenizer, FlaubertTokenizer


tokenizer_types = {'bert': BertTokenizer,
                   'roberta': RobertaTokenizer,
                   'camembert': CamembertTokenizer,
                   'flaubert': FlaubertTokenizer}


TokenizedSentence = tuple[list[int], list[int]]                 # token_ids, cluster_ids
TokenizedSymbol = tuple[int, int, int, int | None]              # token index, positional index, numels, link index
TokenizedTrees = list[list[list[TokenizedSymbol]]]              # level trees
TokenPosition = tuple[int, int, int, int]                       # link index, level index, tree index, position index
TokenizedMatching = list[tuple[TokenPosition, TokenPosition]]   # list of (positive, negative) pairs
TokenizedMatchings = dict[int, [TokenizedMatching]]             # list of matchings, one per atom
TokenizedSample = tuple[TokenizedSentence, TokenizedTrees, TokenizedMatchings | None]


class Tokenizer:
    def __init__(self, core: str, bert_type: str):
        cls = tokenizer_types[bert_type]
        self.core = cls.from_pretrained(core)

    def encode_words(self, words: list[str]) -> TokenizedSentence:
        def f(w: str) -> list[int]:
            return self.core.encode(w, add_special_tokens=False)
        sentence = [(t, i + 1) for i, w in enumerate(words) for t in f(w)]
        sentence = [(self.core.cls_token_id, 0), *sentence, (self.core.sep_token_id, -1)]
        token_ids, cluster_ids = zip(*sentence)
        return list(token_ids), list(cluster_ids)

    def encode_sentence(self, sentence: str) -> tuple[TokenizedSentence, list[str]]:
        words = whitespace_punct(sentence).split()
        return self.encode_words(words), words

    def encode_sample(self, sample: Sample) -> TokenizedSentence:
        return self.encode_words(sample.words)


class AtomTokenizer:
    def __init__(self, symbol_map: dict[int, Symbol], symbol_arities: dict[Symbol, int]):
        self.id_to_token = symbol_map
        self.symbol_arities = symbol_arities
        self.token_to_id = {v: k for k, v in symbol_map.items()}
        self.binaries = {sym for sym, arity in symbol_arities.items() if arity == 2}
        self.zeroaries = {sym for sym, arity in symbol_arities.items() if arity == 0}

    def __len__(self) -> int: return len(self.id_to_token)
    def atom_to_id(self, atom: Symbol) -> int: return self.token_to_id[atom]
    def id_to_atom(self, idx: int) -> Symbol: return self.id_to_token[idx]

    def tokenize_tree(self, tree: Tree[Symbol]) -> Tree[TokenizedSymbol]:
        def go(_tree: Tree[Symbol],
               parent_pos: int) -> Tree[TokenizedSymbol]:
            match _tree:
                case Leaf(atom):
                    return Leaf((self.atom_to_id(atom.plain()), parent_pos, _tree.numel(), atom.index))
                case Binary(node, left, right):
                    left = go(left, 2 * parent_pos)
                    right = go(right, 2 * parent_pos + 1)
                    return Binary((self.atom_to_id(node.plain()), parent_pos, _tree.numel(), None), left, right)
            raise TypeError('Invalid tree type')
        return go(tree, 1)

    def encode_tree(self, tree: Tree[Symbol]) -> list[list[TokenizedSymbol]]:
        return self.tokenize_tree(tree).levels()

    def encode_trees(self, trees: list[Tree[Symbol]]) -> list[list[list[TokenizedSymbol]]]:
        return [self.encode_tree(t) for t in trees]

    def group_indices_by_atom(self, links: dict[Symbol, Symbol]) -> dict[Symbol, [dict[int, int]]]:
        return {atom: {neg.index: pos.index for pos, neg in links.items() if pos.plain() == atom}
                for atom in set(k.plain() for k in links.keys())}

    def encode_links(self,
                     links: dict[Symbol, Symbol],
                     tree_levels: list[list[list[TokenizedSymbol]]]) -> dict[str, TokenizedMatching]:
        index_to_pos = {link_id: (link_id, level_id, root_id, pos_id)
                        for root_id, tree in enumerate(tree_levels)
                        for level_id, level in enumerate(tree)
                        for (_, pos_id, _, link_id) in level
                        if link_id is not None}

        def locate_index(link_index: int) -> TokenPosition:
            return index_to_pos[link_index]

        grouped_links = self.group_indices_by_atom(links)
        return {self.atom_to_id(atom): [(locate_index(neg), locate_index(pos)) for neg, pos in atom_links.items()]
                for atom, atom_links in grouped_links.items()}

    def levels_to_trees(self, node_ids: list[list[int]]) -> list[Tree[Symbol]]:
        levels = [[self.id_to_token[i] for i in level] for level in node_ids]
        fringe: list[Tree[Symbol]] = [Leaf(symbol) for symbol in levels[-1]]
        for level in reversed(levels[:-1]):
            stack = list(reversed(fringe))
            fringe = []
            for symbol in level:
                if self.symbol_arities[symbol] == 2:
                    left = stack.pop()
                    right = stack.pop()
                    fringe.append(Binary(symbol, left, right))
                else:
                    fringe.append(Leaf(symbol))
        return fringe

    def levels_to_ptrees(self, node_ids: list[list[int]]) -> list[Tree[tuple[Symbol, tuple[int, int]]]]:
        # same as above, but tree nodes remember their position in the decoder
        levels = [[(l_idx, n_idx, self.id_to_token[n_id]) for n_idx, n_id in enumerate(level)]
                  for l_idx, level in enumerate(node_ids)]
        fringe: list[Tree[tuple[Symbol, tuple[int, int]]]]
        fringe = [Leaf((s, (l_idx, n_idx))) for l_idx, n_idx, s in levels[-1]]
        for level in reversed(levels[:-1]):
            stack = list(reversed(fringe))
            fringe = []
            for (l_idx, n_idx, s) in level:
                if self.symbol_arities[s] == 2:
                    left = stack.pop()
                    right = stack.pop()
                    fringe.append(Binary((s, (l_idx, n_idx)), left, right))
                else:
                    fringe.append(Leaf((s, (l_idx, n_idx))))
        return fringe

    @staticmethod
    def from_file(file_path: str) -> AtomTokenizer:
        id_to_sym, sym_to_arity = {}, {}
        with open(file_path, 'r') as f:
            for line in f.readlines():
                idx, name, arity = line.split('\t')
                id_to_sym[eval(idx)] = (symbol := Symbol(name))
                sym_to_arity[symbol] = eval(arity)
        return AtomTokenizer(id_to_sym, sym_to_arity)


def index_ptree(tree: Tree[tuple[Symbol, tuple[int, int]]],
                index: int,
                ignoring: set[str]) -> tuple[Tree[tuple[Symbol, tuple[int, int]]], int]:
    match tree:
        case Leaf((Symbol(name), p)):
            if name in ignoring:
                return Leaf((Symbol(name), p)), index
            return Leaf((Symbol(name, index), p)), index + 1
        case Unary(node, content):
            content, index = index_ptree(content, index, ignoring)
            return Unary(node, content), index
        case Binary(node, left, right):
            left, index = index_ptree(left, index, ignoring)
            right, index = index_ptree(right, index, ignoring)
            return Binary(node, left, right), index
        case _:
            raise ValueError(f'Unknown tree type: {tree}')


def index_ptrees(*trees: Tree[tuple[Symbol, tuple[int, int]]],
                 ignoring: set[str]) -> list[Tree[tuple[Symbol, tuple[int, int]]]]:
    ret, index = [], 0
    for tree in trees:
        indexed, index = index_ptree(tree, index, ignoring)
        ret.append(indexed)
    return ret


def group_trees(trees: list[Tree[Node]], splitpoints: list[int]) -> list[list[Tree[Node]]]:
    return [trees[start:end] for start, end in zip([0] + splitpoints, splitpoints)]


def encode_sample(
        sample: Sample,
        atokenizer: AtomTokenizer,
        tokenizer: Tokenizer) -> TokenizedSample:
    encoder_inputs = tokenizer.encode_sample(sample)
    decoder_inputs = atokenizer.encode_trees(sample.trees)
    if sample.links is not None:
        parser_inputs = atokenizer.encode_links(sample.links, decoder_inputs)
    else:
        parser_inputs = None
    return encoder_inputs, decoder_inputs, parser_inputs


def tokenize_dataset(data: tuple[list[Sample], ...],
                     atom_map_path: str,
                     bert_name: str,
                     bert_type: str) -> tuple[list[TokenizedSample], ...]:
    atokenizer = AtomTokenizer.from_file(atom_map_path)
    tokenizer = Tokenizer(bert_name, bert_type)
    return tuple([encode_sample(s, atokenizer, tokenizer) for s in subset] for subset in data)
