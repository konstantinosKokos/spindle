from __future__ import annotations

import pdb

import torch
from torch.nn import Module, Linear, Dropout, Parameter, LeakyReLU
from torch.nn.utils.parametrizations import orthogonal
from math import sqrt
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax as sparse_softmax
from .utils import SwiGLU, RMSNorm, swish


class Decoder(Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, cross_heads: int, self_heads: int, dropout_rate: float):
        super().__init__()
        self.nodes_to_root = GCATv2Conv(encoder_dim, decoder_dim, cross_heads, dropout_rate)
        self.roots_to_root = SelfMHA(encoder_dim, self_heads)
        self._ffn = SwiGLU(encoder_dim, int(8 / 3 * encoder_dim))
        self.fringe_gate = Linear(decoder_dim, decoder_dim)
        self.root_to_fringe = Linear(encoder_dim, decoder_dim)
        self.node_feedback_norm = RMSNorm(encoder_dim)
        self.root_feedback_norm = RMSNorm(encoder_dim)
        self.ffn_norm = RMSNorm(encoder_dim)
        self.dropout = Dropout(dropout_rate)

    def node_feedback(self, root_features: Tensor, node_features: OptTensor, root_index: OptTensor) -> Tensor:
        if node_features is None:
            return self.node_feedback_norm(root_features)
        feedback = self.nodes_to_root(xs=root_features, ctx=node_features, edge_index=root_index)
        return self.node_feedback_norm(root_features + feedback)

    def decode_fringe(self, root_features: Tensor, fringe_features: Tensor, root_index: Tensor) -> Tensor:
        fringe_gates = swish(self.fringe_gate(fringe_features))
        root_to_fringe = self.root_to_fringe(root_features[root_index])
        return root_to_fringe * fringe_gates

    def root_feedback(self, root_features: Tensor, root_edge_index: Tensor, root_edge_attr: Tensor) -> Tensor:
        feedback = self.roots_to_root(xs=root_features, edge_index=root_edge_index, edge_attr=root_edge_attr)
        return self.root_feedback_norm(root_features + feedback)

    def ffn(self, root_features: Tensor) -> Tensor:
        ffn = self._ffn(root_features)
        return self.ffn_norm(root_features + ffn)

    def step(self,
             root_features: Tensor,
             feedback_features: OptTensor,
             feedback_index: OptTensor,
             fringe_features: Tensor,
             root_to_fringe_index: Tensor,
             root_edge_index: Tensor,
             root_edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        root_features = self.node_feedback(root_features, feedback_features, feedback_index)
        root_features = self.dropout(root_features)
        root_features = self.root_feedback(root_features, root_edge_index, root_edge_attr)
        root_features = self.dropout(root_features)
        root_features = self.ffn(root_features)
        root_features = self.dropout(root_features)
        fringe_features = self.decode_fringe(root_features, fringe_features, root_to_fringe_index)
        return root_features, fringe_features


class SelfMHA(MessagePassing):
    def __init__(self, dim: int, num_heads: int):
        super(SelfMHA, self).__init__(aggr="add", node_dim=0)
        assert dim % num_heads == 0
        self.w_qkv = Linear(dim, 3 * dim, bias=False)
        self.num_heads = num_heads
        self.dim = dim
        self.hdim = dim // num_heads

    def forward(self, xs: Tensor, edge_index: Tensor, edge_attr: Tensor):
        qs, ks, vs = self.w_qkv(xs).view(-1, self.num_heads, self.hdim, 3).chunk(3, dim=-1)
        qs, ks, vs = qs.squeeze(-1), ks.squeeze(-1), vs.squeeze(-1)
        return self.propagate(edge_index, qs=qs, ks=ks, vs=vs, edge_attr=edge_attr).view(-1, self.dim)

    def message(self, qs_i: Tensor, ks_j: Tensor, vs_j: Tensor, edge_attr: Tensor, index: Tensor):
        atn = (qs_i * ks_j * edge_attr.unsqueeze(1)).sum(dim=-1) / sqrt(self.hdim)
        atn = sparse_softmax(atn, index, dim=0)
        return atn.unsqueeze(-1) * vs_j


class GCATv2Conv(MessagePassing):
    def __init__(self, self_dim: int, ctx_dim: int,  num_heads: int, negative_slope: float = 0.2):
        super(GCATv2Conv, self).__init__(aggr='add', node_dim=0)
        self.num_heads = num_heads
        self.hdim = self_dim // num_heads
        self.dim = self_dim
        self.ctx_to_x = Linear(ctx_dim, self_dim, bias=False)
        self.x_to_g = Linear(self_dim, self.num_heads, bias=False)
        self.attn = Linear(ctx_dim + self_dim, self.num_heads, bias=False)
        self.relu = LeakyReLU(negative_slope)

    def forward(self, xs: Tensor, ctx: Tensor, edge_index: Tensor):
        messages = self.propagate(xs=xs, ctx=ctx, edge_index=edge_index)
        residual_gates = torch.ones(xs.shape[0], self.num_heads, 1, device=xs.device)
        residual_indices = edge_index[1].unique()
        residual_gates[residual_indices] = self.x_to_g(xs[residual_indices]).unsqueeze(-1)
        out = residual_gates * xs.view(-1, self.num_heads, self.hdim) + (1 - residual_gates) * messages
        return out.view(-1, self.dim)

    def message(self, xs_i: Tensor, ctx_j: Tensor, index: Tensor):
        alphas = self.relu(self.attn(torch.cat((ctx_j, xs_i), dim=-1)))
        alphas = sparse_softmax(alphas, index, dim=0)
        return alphas.unsqueeze(-1) * self.ctx_to_x(ctx_j).view(-1, self.num_heads, self.hdim)


class BinaryPathEncoder(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.primitives = Parameter(torch.nn.init.normal_(torch.empty(2, dim, dim)))
        self.dim = dim
        self.precomputed = None

    def precompute(self, up_to: int):
        precomputed = [self.primitives[0],
                       self.primitives[1]]
        for i in range(3, up_to + 1):
            precomputed.append(precomputed[i//2] @ precomputed[1 - i % 2])
        self.precomputed = torch.stack(precomputed)

    def forward(self, node_positions: Tensor) -> Tensor:
        return torch.index_select(self.precomputed, 0, node_positions - 1)

    @staticmethod
    def orthogonal(dim: int) -> BinaryPathEncoder:
        return orthogonal(BinaryPathEncoder(dim), name='primitives', orthogonal_map='matrix_exp')  # type: ignore
