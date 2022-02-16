# todo: assert homogeneous tokenization in encode sentence (inference) and encode words (training)
from __future__ import annotations

from .preprocessing import Sample, Symbol, whitespace_punct
from .tree import Tree, Leaf, Binary


import torch
from torch import Tensor
from transformers import BertTokenizer


class Tokenizer:
    def __init__(self, core: str):
        self.core = BertTokenizer.from_pretrained(core)

    def encode_words(self, words: list[str]) -> tuple[Tensor, Tensor]:
        subword_tokens = [[self.core.cls_token_id]] + \
                         [self.core.encode(w, add_special_tokens=False) for w in words] + \
                         [[self.core.sep_token_id]]
        word_clusters = [[i] * len(subword_tokens[i]) for i in range(len(subword_tokens) - 1)] + [[-1]]
        return torch.tensor(sum(subword_tokens, [])), torch.tensor(sum(word_clusters, []))

    def encode_sentence(self, sentence: str) -> tuple[Tensor, Tensor]:
        return self.encode_words(whitespace_punct(sentence).split())

    def encode_sample(self, sample: Sample) -> tuple[Tensor, Tensor]:
        return self.encode_words(sample.words)


class AtomTokenizer:
    def __init__(self, symbol_map: dict[int, Symbol], symbol_arities: dict[Symbol, int]):
        self.id_to_token = symbol_map
        self.symbol_arities = symbol_arities
        self.token_to_id = {v: k for k, v in symbol_map.items()}
        self.mask_token_id = self.token_to_id[Symbol('[MASK]')]
        self.binaries = {sym for sym, arity in symbol_arities.items() if arity == 2}
        self.zeroaries = {sym for sym, arity in symbol_arities.items() if arity == 0}

    def __len__(self) -> int: return len(self.id_to_token)
    def atom_to_id(self, atom: Symbol) -> int: return self.token_to_id[atom]
    def id_to_atom(self, idx: int) -> Symbol: return self.id_to_token[idx]

    def positionally_encode_tree(self, tree: Tree[Symbol]) -> Tree[tuple[int, int]]:
        def f(_tree: Tree[Symbol], parent: int) -> Tree[tuple[int, int]]:
            match _tree:
                case Leaf(atom):
                    return Leaf((self.token_to_id[atom.plain()], parent))
                case Binary(atom, left, right):
                    return Binary((self.token_to_id[atom.plain()], parent),
                                  f(left, 2 * parent),
                                  f(right, 2 * parent + 1))
                case _:
                    raise TypeError(f'Unsupported tree type: {_tree}')
        return f(tree, 1)

    def encode_tree(self, tree: Tree[Symbol]) -> list[list[tuple[int, int]]]:
        return self.positionally_encode_tree(tree).levels()

    def encode_trees(self, trees: list[Tree[Symbol]]) -> list[list[list[tuple[int, int]]]]:
        return [self.encode_tree(t) for t in trees]

    def encode_sample(self, sample: Sample) -> list[list[list[tuple[int, int]]]]:
        return self.encode_trees(sample.trees)

    @staticmethod
    def from_file(file_path: str) -> AtomTokenizer:
        id_to_sym, sym_to_arity = {}, {}
        with open(file_path, 'r') as f:
            for line in f.readlines():
                idx, name, arity = line.split('\t')
                id_to_sym[eval(idx)] = (symbol := Symbol(name))
                sym_to_arity[symbol] = eval(arity)
        return AtomTokenizer(id_to_sym, sym_to_arity)


def encode_sample(
        sample: Sample,
        atokenizer: AtomTokenizer,
        tokenizer: Tokenizer) -> tuple[tuple[Tensor, Tensor], list[list[list[tuple[int, int]]]]]:
    return tokenizer.encode_sample(sample), atokenizer.encode_trees(sample.trees)


def vectorize(data: tuple[list[Sample], list[Sample], list[Sample]],
              atom_map_path: str,
              bert_name: str) \
        -> tuple[list[tuple[tuple[Tensor, Tensor], list[list[list[tuple[int, int]]]]]],
                 list[tuple[tuple[Tensor, Tensor], list[list[list[tuple[int, int]]]]]],
                 list[tuple[tuple[Tensor, Tensor], list[list[list[tuple[int, int]]]]]]]:
    atoken = AtomTokenizer.from_file(atom_map_path)
    tokenizer = Tokenizer(bert_name)
    train, dev, test = data
    return ([encode_sample(s, atoken, tokenizer) for s in train],
            [encode_sample(s, atoken, tokenizer) for s in dev],
            [encode_sample(s, atoken, tokenizer) for s in test])