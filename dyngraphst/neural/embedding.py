import pdb

import torch
from torch.nn import Module, Dropout, Parameter
from torch.nn.functional import linear, embedding
from torch import Tensor


class InvertibleEmbedding(Module):
    def __init__(self, num_classes: int, dim: int, dropout_rate: float):
        super(InvertibleEmbedding, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.dropout = Dropout(dropout_rate)
        self.weight = Parameter(torch.nn.init.normal_(torch.empty(num_classes, dim)), requires_grad=True)

    def embed(self, xs: Tensor) -> Tensor:
        return self.dropout(embedding(xs, self.weight))

    def invert(self, xs: Tensor) -> Tensor:
        return linear(xs, self.weight)
