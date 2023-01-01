from .tree_decoding import Decoder, BinaryPathEncoder
from .encoder import Encoder
from .embedding import InvertibleEmbedding
from .linker import Linker

import torch
from torch.nn import Module, Embedding, Dropout
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.utils import dropout_adj


class Parser(Module):
    def __init__(self,
                 num_classes: int,
                 max_dist: int,
                 encoder_config_or_name: str | dict,
                 bert_type: str,
                 sep_token_id: int,
                 encoder_dim: int = 768,
                 decoder_dim: int = 128,
                 cross_heads: int = 4,
                 self_heads: int = 8,
                 dropout_rate: float = 0.2,
                 edge_dropout: float = 0.2,
                 truncate_long_edges: bool = True, ):
        super(Parser, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.max_dist = max_dist
        self.edge_dropout = edge_dropout
        self.truncate_long_edges = truncate_long_edges

        self.encoder = Encoder(encoder_config_or_name, bert_type, sep_token_id, encoder_dim)
        self.decoder = Decoder(encoder_dim, decoder_dim, cross_heads, self_heads)
        self.dropout = Dropout(dropout_rate)
        self.path_encoder = BinaryPathEncoder.orthogonal(self.decoder_dim)
        self.embedder = InvertibleEmbedding(num_classes, decoder_dim)
        self.dist_embedding = Embedding(2 * max_dist + 2, encoder_dim // self_heads)
        self.linker = Linker(self.decoder_dim, dropout_rate)

        self.imprint = f'\
                       \tencoder_core: {encoder_config_or_name}\n\
                       \tencoder_dim: {encoder_dim}\n\
                       \tdecoder_dim: {decoder_dim}\n\
                       \tcross_heads: {cross_heads}\n\
                       \tself_heads: {self_heads}\n\
                       \tdropout_rate: {dropout_rate}\n\
                       \tedge_dropout: {edge_dropout}\n\
                       \ttruncate_long_edges: {truncate_long_edges}\n'

    def encode(self, input_ids: Tensor, attention_mask: Tensor, token_clusters: Tensor) -> Tensor:
        return self.encoder.forward(input_ids, attention_mask, token_clusters)

    def decode_step(self,
                    root_features: Tensor,
                    feedback_features: OptTensor,
                    fringe_maps: Tensor,
                    feedback_index: OptTensor,
                    root_edge_index: Tensor,
                    root_to_fringe_index: Tensor,
                    root_edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        return self.decoder.step(root_features=root_features,
                                 feedback_features=feedback_features,
                                 fringe_maps=fringe_maps,
                                 feedback_index=feedback_index,
                                 root_edge_index=root_edge_index,
                                 root_to_fringe_index=root_to_fringe_index,
                                 root_edge_attr=root_edge_attr)

    def decode_train(self,
                     root_features: Tensor,
                     node_ids: list[Tensor],
                     node_pos: list[Tensor],
                     root_to_node_index: list[Tensor],
                     root_edge_index: Tensor,
                     root_edge_attr: Tensor) -> tuple[list[Tensor], list[Tensor]]:

        root_features = self.dropout(root_features)

        feedback_features, feedback_index, preds, decoder_reprs = None, None, [], []
        self.path_encoder.precompute(max(node_pos[-1]))
        for i in range(len(node_ids)):
            positional_maps = self.path_encoder.forward(node_pos[i])
            root_features, fringe_weights = self.decode_step(root_features=root_features,
                                                             feedback_features=feedback_features,
                                                             fringe_maps=positional_maps,
                                                             feedback_index=feedback_index,
                                                             root_edge_index=root_edge_index,
                                                             root_edge_attr=root_edge_attr,
                                                             root_to_fringe_index=root_to_node_index[i][0])
            fringe_weights = self.dropout(fringe_weights)
            fringe_logits = self.embedder.invert(fringe_weights)
            feedback_features = self.positionally_embed(positional_maps, node_ids[i])
            feedback_index = root_to_node_index[i].flip(0)
            preds.append(fringe_logits)
            decoder_reprs.append(fringe_weights)
        return preds, decoder_reprs

    @torch.no_grad()
    def decode_dev(self,
                   root_features: Tensor,
                   root_edge_index: Tensor,
                   root_edge_attr: Tensor,
                   max_type_depth: int,
                   first_binary: int) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        fringe_pos = torch.ones(root_features.shape[0], device=root_features.device, dtype=torch.long)
        root_to_fringe_index = torch.arange(root_features.shape[0], device=root_features.device)

        feedback_features, feedback_index, preds, decoder_reprs, node_pos = None, None, [], [], [fringe_pos]
        for _ in range(max_type_depth):
            positional_maps = self.path_encoder.forward(fringe_pos)
            root_features, fringe_weights = self.decode_step(root_features=root_features,
                                                             feedback_features=feedback_features,
                                                             fringe_maps=positional_maps,
                                                             feedback_index=feedback_index,
                                                             root_edge_index=root_edge_index,
                                                             root_edge_attr=root_edge_attr,
                                                             root_to_fringe_index=root_to_fringe_index)
            node_ids = self.embedder.invert(fringe_weights).argmax(dim=-1)
            binary_mask = node_ids >= first_binary

            feedback_features = self.positionally_embed(positional_maps, node_ids)
            feedback_index = torch.vstack(
                (torch.arange(root_to_fringe_index.shape[0], device=feedback_features.device),
                 root_to_fringe_index))

            root_to_fringe_index = root_to_fringe_index[binary_mask].repeat_interleave(2)
            left_pos = 2 * fringe_pos[binary_mask]
            right_pos = left_pos + 1
            fringe_pos = torch.stack((left_pos, right_pos), dim=1).view(-1)

            preds.append(node_ids)
            decoder_reprs.append(fringe_weights)
            node_pos.append(fringe_pos)
            if not binary_mask.any():
                return preds, decoder_reprs, node_pos
        return preds, decoder_reprs, node_pos

    def forward_tagger_train(self,
                             input_ids: Tensor,
                             attention_mask: Tensor,
                             token_clusters: Tensor,
                             node_ids: list[Tensor],
                             node_pos: list[Tensor],
                             node_to_root_index: list[Tensor],
                             root_edge_index: Tensor,
                             root_dist: Tensor,
                             cls_dist: int = -999) -> tuple[list[Tensor], list[Tensor]]:
        root_features = self.encode(input_ids, attention_mask, token_clusters)
        root_edge_index, root_edge_attr = self.embed_sentential_edges(root_edge_index, root_dist, cls_dist)
        token_preds, decoder_reprs = self.decode_train(root_features=root_features,
                                                       node_ids=node_ids,
                                                       node_pos=node_pos,
                                                       root_to_node_index=node_to_root_index,
                                                       root_edge_index=root_edge_index,
                                                       root_edge_attr=root_edge_attr)
        return token_preds, decoder_reprs

    def forward_train(self,
                      input_ids: Tensor,
                      attention_mask: Tensor,
                      token_clusters: Tensor,
                      node_ids: list[Tensor],
                      node_pos: list[Tensor],
                      node_to_root_index: list[Tensor],
                      root_edge_index: Tensor,
                      root_dist: Tensor,
                      link_indices: list[Tensor],
                      cls_dist: int = -999) -> tuple[list[Tensor], list[Tensor]]:
        token_preds, decoder_reprs = self.forward_tagger_train(input_ids=input_ids,
                                                               attention_mask=attention_mask,
                                                               token_clusters=token_clusters,
                                                               node_ids=node_ids,
                                                               node_pos=node_pos,
                                                               node_to_root_index=node_to_root_index,
                                                               root_edge_index=root_edge_index,
                                                               root_dist=root_dist,
                                                               cls_dist=cls_dist)
        matches = self.link(decoder_reprs, link_indices)
        return token_preds, matches

    def forward_dev(self,
                    input_ids: Tensor,
                    attention_mask: Tensor,
                    token_clusters: Tensor,
                    root_edge_index: Tensor,
                    root_dist: Tensor,
                    first_binary: int,
                    cls_dist: int = -999,
                    max_type_depth: int = 10) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        root_features = self.encode(input_ids, attention_mask, token_clusters)
        root_edge_index, root_edge_attr = self.embed_sentential_edges(root_edge_index, root_dist, cls_dist)
        return self.decode_dev(root_features=root_features,
                               root_edge_index=root_edge_index,
                               root_edge_attr=root_edge_attr,
                               first_binary=first_binary,
                               max_type_depth=max_type_depth)

    def embed_sentential_edges(self, root_edge_index: Tensor, root_dist: Tensor, cls_dist) -> tuple[Tensor, Tensor]:
        clipped_dist = torch.where(root_dist != cls_dist,
                                   root_dist.clip(-self.max_dist, self.max_dist) + self.max_dist + 1,
                                   0)
        if self.truncate_long_edges:
            distance_mask = root_dist.eq(cls_dist).bitwise_or(root_dist.abs() <= self.max_dist)
            clipped_dist = clipped_dist[distance_mask]
            root_edge_index = root_edge_index[:, distance_mask]

        root_edge_index, clipped_dist = dropout_adj(
            root_edge_index, clipped_dist, self.edge_dropout, training=self.training)
        return root_edge_index, self.dist_embedding(clipped_dist)

    def positionally_embed(self, positional_maps: Tensor, node_ids: Tensor) -> Tensor:
        return positional_maps * self.embedder.embed(node_ids)

    def link(self, reprs: list[Tensor],
             indices: list[Tensor],
             training: bool = True,
             num_iters: int = 3) -> list[Tensor]:
        return self.linker(reprs, indices, training=training, num_iters=num_iters)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: torch.device = 'cpu', strict: bool = False):
        self.load_state_dict(torch.load(path, map_location=map_location), strict=strict)
