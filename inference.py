import torch
from torch import Tensor

from dyngraphpn.neural.model import Parser
from dyngraphpn.data.tokenization import (Tokenizer, AtomTokenizer, Leaf, Symbol, Tree,
                                          group_trees, index_ptrees)
from dyngraphpn.data.processing import merge_on_word_starts, get_word_starts
from dyngraphpn.neural.batching import batchify_encoder_inputs, ptrees_to_candidates, BackPointer
from dyngraphpn.data.aethel_interface import (tree_to_ft, links_to_proof, ft_to_type,
                                              LexicalPhrase, LexicalItem, Proof,
                                              AxiomLinks, FormulaTree)

from dataclasses import dataclass
from transformers import BertConfig
from itertools import accumulate

from scipy.optimize import linear_sum_assignment


@dataclass
class Analysis:
    lexical_phrases:   tuple[LexicalPhrase, ...]
    proof:             Proof | Exception

    @property
    def sentence(self):
        return ' '.join(phrase.string for phrase in self.lexical_phrases)


class InferenceWrapper:
    def __init__(self,
                 weight_path: str,
                 atom_map_path: str = './data/atom_map.tsv',
                 config_path: str | None = './data/bert_config.json',
                 device: torch.device = 'cuda'):
        encoder = 'GroNLP/bert-base-dutch-cased' if config_path is None else BertConfig.from_json_file(config_path)
        self.parser = Parser(num_classes=81,
                             max_dist=6,
                             encoder_config_or_name=encoder,
                             bert_type='bert',
                             sep_token_id=2).to(device)
        self.tokenizer = Tokenizer(core='GroNLP/bert-base-dutch-cased', bert_type='bert')
        self.atom_tokenizer = AtomTokenizer.from_file(atom_map_path)
        self.parser.load(weight_path, map_location=device, strict=True)
        self.parser.eval()
        self.parser.path_encoder.precompute(2 ** 16)
        self.device = device
        self.first_binary = next(k for k, v in sorted(self.atom_tokenizer.id_to_token.items(), key=lambda kv: kv[0])
                                 if self.atom_tokenizer.symbol_arities[v] == 2)

    @torch.no_grad()
    def analyze(self, sentences: list[str]) -> list[Analysis]:
        tokenized_sents, split_sents = zip(*map(self.tokenizer.encode_sentence, sentences))
        token_ids, cluster_ids = zip(*tokenized_sents)
        encoder_batch, sent_lens = batchify_encoder_inputs(token_ids=token_ids,
                                                           token_clusters=cluster_ids,
                                                           pad_token_id=self.tokenizer.core.pad_token_id)
        encoder_batch = encoder_batch.to(self.device)
        (node_ids, decoder_reprs, node_pos) \
            = self.parser.forward_dev(input_ids=encoder_batch.token_ids,
                                      attention_mask=encoder_batch.atn_mask,
                                      token_clusters=encoder_batch.cluster_ids,
                                      root_edge_index=encoder_batch.edge_index,
                                      root_dist=encoder_batch.edge_attr,
                                      first_binary=self.first_binary,
                                      max_type_depth=16)
        groups = group_trees(self.atom_tokenizer.levels_to_ptrees([n.tolist() for n in node_ids]),
                             list(accumulate(sent_lens)))
        analyses: list[Analysis] = []
        for i, (words, ptrees) in enumerate(zip(split_sents, groups)):
            words, ptrees = merge_and_index(words, ptrees)
            f_conclusion, f_assignments = ptrees_to_formulas(ptrees)
            lex_phrases = make_lex_phrases(words, f_assignments)
            if (candidates := ptrees_to_candidates(ptrees)) is not None:
                grouped_matches = self.parser.link(decoder_reprs, candidates.indices, training=False, num_iters=3)
                links = matches_to_links(grouped_matches, candidates.backpointers)
                proof = attempt_traversal(links, f_assignments, f_conclusion)
            else:
                proof = ValueError('Invariance check failed.')
            analyses.append(Analysis(lexical_phrases=lex_phrases, proof=proof))
        return analyses


def merge_and_index(words: list[str], ptrees: list[Tree[tuple[Symbol, tuple[int, int]]]]) -> \
        tuple[list[str], list[Tree[tuple[Symbol, tuple[int, int]]]]]:
    (conclusion, *assignments) = ptrees
    word_starts = get_word_starts([tree.fmap(lambda node: node[0]) for tree in assignments])
    words, assignments = merge_on_word_starts(words, assignments, word_starts)
    ptrees = index_ptrees(*[conclusion, *assignments], ignoring={'PUNCT'})
    return words, ptrees


def matches_to_links(grouped_matches: list[Tensor], backpointers: list[BackPointer]) -> AxiomLinks:
    links: AxiomLinks = {}

    def sign_to_ft(s: str, idx: int, pol: bool) -> FormulaTree: return tree_to_ft(Leaf(Symbol(s, idx)), pol)

    def solve(x: Tensor) -> list[int]:
        discretized = x.argmax(dim=-1).tolist()
        if len(set(discretized)) == len(discretized):
            return discretized
        return linear_sum_assignment(x, maximize=True)[1].tolist()

    for match_tensor, backpointer_group in zip(grouped_matches, backpointers):
        matches = [solve(m) for m in match_tensor.exp().cpu()]
        for match, (_, atom, neg_indices, pos_indices) in zip(matches, backpointer_group):
            links |= {sign_to_ft(atom, neg_indices[i], False): sign_to_ft(atom, pos_indices[match[i]], True)
                      for i in range(len(match))}

    return links


def ptrees_to_formulas(ptrees: list[Tree[tuple[Symbol, tuple[int, int]]]]) -> tuple[FormulaTree, list[FormulaTree]]:
    (conclusion, *assignments) = ptrees
    f_assignments = [tree_to_ft(tree.fmap(lambda node: node[0]), True) for tree in assignments]
    f_conclusion = tree_to_ft(conclusion.fmap(lambda node: node[0]), False)
    return f_conclusion, f_assignments


def make_lex_phrases(words: list[str], assignments: list[FormulaTree]) -> tuple[LexicalPhrase, ...]:
    return tuple(LexicalPhrase(tuple(LexicalItem(word, 'None', 'None', 'None') for word in phrase.split()),
                               ft_to_type(assignment))
                 for phrase, assignment in zip(words, assignments))


def attempt_traversal(links: AxiomLinks,
                      assignments: list[FormulaTree],
                      conclusion: FormulaTree) -> Proof | Exception:
    try:
        return links_to_proof(links, {i: f for i, f in enumerate(assignments)}, conclusion)
    except Exception as e:
        return e
