import pdb

from .tree_decoding import Decoder, BinaryPathEncoder
from .encoder import Encoder
from .embedding import InvertibleEmbedding

import torch
from torch.nn import Module, Embedding, Linear
from torch import Tensor
from torch_geometric.typing import OptTensor


class Tagger(Module):
    def __init__(self,
                 num_classes: int,
                 max_dist: int,
                 encoder_core: str,
                 sep_token_id: int,
                 encoder_dim: int = 768,
                 decoder_dim: int = 128,
                 cross_heads: int = 4,
                 self_heads: int = 8,
                 dropout_rate: float = 0.15):
        super(Tagger, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.max_dist = max_dist
        self.encoder = Encoder(encoder_core, sep_token_id, dropout_rate)
        self.encoder_out = Linear(self.encoder_dim, self.encoder_dim)
        self.decoder = Decoder(encoder_dim, decoder_dim, cross_heads, self_heads, dropout_rate)
        self.path_encoder = BinaryPathEncoder.orthogonal(self.decoder_dim)
        self.embedder = InvertibleEmbedding(num_classes, decoder_dim, dropout_rate)
        self.dist_embedding = Embedding(2 * max_dist + 2, encoder_dim // self_heads)

    def encode(self,
               input_ids: Tensor,
               attention_mask: Tensor,
               token_clusters: Tensor) -> Tensor:
        bert_out = self.encoder.forward(input_ids, attention_mask, token_clusters)
        return self.encoder_out(bert_out)

    def decode_step(self,
                    root_features: Tensor,
                    feedback_features: OptTensor,
                    fringe_features: Tensor,
                    feedback_index: OptTensor,
                    root_edge_index: Tensor,
                    root_to_fringe_index: Tensor,
                    root_edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        return self.decoder.step(root_features=root_features,
                                 feedback_features=feedback_features,
                                 fringe_features=fringe_features,
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
                     root_edge_attr: Tensor,
                     mask_token_id: int = 0) -> list[Tensor]:
        feedback_features, feedback_index, preds = None, None, []
        self.path_encoder.precompute(max(node_pos[-1]))
        for i in range(len(node_ids)):
            positional_maps = self.path_encoder.forward(node_pos[i])
            fringe_features = self.embedder.embed(torch.ones_like(node_ids[i]) * mask_token_id)
            fringe_features = torch.bmm(positional_maps, fringe_features.unsqueeze(-1)).squeeze(-1)
            root_features, fringe_weights = self.decode_step(root_features=root_features,
                                                             feedback_features=feedback_features,
                                                             fringe_features=fringe_features,
                                                             feedback_index=feedback_index,
                                                             root_edge_index=root_edge_index,
                                                             root_edge_attr=root_edge_attr,
                                                             root_to_fringe_index=root_to_node_index[i][0])
            feedback_features = self.embedder.embed(node_ids[i])
            feedback_features = torch.bmm(positional_maps, feedback_features.unsqueeze(-1)).squeeze(-1)
            feedback_index = root_to_node_index[i].flip(0)
            preds.append(self.embedder.invert(fringe_weights))
        return preds

    @torch.no_grad()
    def decode_dev(self,
                   root_features: Tensor,
                   root_edge_index: Tensor,
                   root_edge_attr: Tensor,
                   max_type_depth: int,
                   mask_token_id: int = 0,
                   splitpoint: int = 32):

        fringe_pos = torch.ones(root_features.shape[0], device=root_features.device, dtype=torch.long)
        fringe_ids = torch.ones_like(fringe_pos, dtype=torch.long) * mask_token_id
        root_to_fringe_index = torch.arange(root_features.shape[0], device=root_features.device)

        feedback_features, feedback_index, preds = None, None, []

        for _ in range(max_type_depth):
            positional_maps = self.path_encoder.forward(fringe_pos)
            fringe_features = self.positionally_embed(positional_maps, fringe_ids)

            root_features, fringe_weights = self.decode_step(root_features=root_features,
                                                             feedback_features=feedback_features,
                                                             fringe_features=fringe_features,
                                                             feedback_index=feedback_index,
                                                             root_edge_index=root_edge_index,
                                                             root_edge_attr=root_edge_attr,
                                                             root_to_fringe_index=root_to_fringe_index)
            node_ids = self.embedder.invert(fringe_weights).argmax(dim=-1)
            binary_mask = node_ids >= splitpoint

            feedback_features = self.positionally_embed(positional_maps, node_ids)
            feedback_index = torch.vstack((torch.arange(root_to_fringe_index.shape[0], device=fringe_ids.device),
                                           root_to_fringe_index))

            root_to_fringe_index = root_to_fringe_index[binary_mask].repeat_interleave(2)
            left_pos = 2 * fringe_pos[binary_mask]
            right_pos = left_pos + 1
            fringe_pos = torch.stack((left_pos, right_pos), dim=1).view(-1)
            fringe_ids = torch.ones_like(fringe_pos, dtype=torch.long) * mask_token_id

            preds.append(node_ids)
            if not binary_mask.any():
                break
        return preds

    def forward_train(self, input_ids: Tensor,
                      attention_mask: Tensor,
                      token_clusters: Tensor,
                      node_ids: list[Tensor],
                      node_pos: list[Tensor],
                      node_to_root_index: list[Tensor],
                      root_to_root_index: Tensor,
                      root_to_root_dist: Tensor,
                      cls_dist: int = -100) -> list[Tensor]:
        root_features = self.encode(input_ids, attention_mask, token_clusters)
        root_dist = torch.where(root_to_root_dist != cls_dist,
                                root_to_root_dist.clip(-self.max_dist, self.max_dist) + self.max_dist + 1,
                                0)
        root_edge_attr = self.dist_embedding(root_dist)
        return self.decode_train(root_features,
                                 node_ids,
                                 node_pos,
                                 node_to_root_index,
                                 root_to_root_index,
                                 root_edge_attr)

    def forward_dev(self, input_ids: Tensor,
                    attention_mask: Tensor,
                    token_clusters: Tensor,
                    root_to_root_index: Tensor,
                    root_to_root_dist: Tensor,
                    cls_dist: int = -100,
                    max_type_depth: int = 10) -> list[Tensor]:
        root_features = self.encode(input_ids, attention_mask, token_clusters)
        root_dist = torch.where(root_to_root_dist != cls_dist,
                                root_to_root_dist.clip(-self.max_dist, self.max_dist) + self.max_dist + 1,
                                0)
        root_edge_attr = self.dist_embedding(root_dist)
        return self.decode_dev(root_features,
                               root_to_root_index,
                               root_edge_attr,
                               max_type_depth=max_type_depth)

    def positionally_embed(self, positional_maps: Tensor, node_ids: Tensor) -> Tensor:
        node_features = self.embedder.embed(node_ids)
        return torch.bmm(positional_maps, node_features.unsqueeze(-1)).squeeze(-1)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str = 'cpu', strict: bool = False):
        self.load_state_dict(torch.load(path, map_location=map_location), strict=strict)
