import torch
from torch import Tensor, logsumexp
from torch.nn import Module, Dropout
from opt_einsum import contract


class Linker(Module):
    def __init__(self, dim: int, dropout_rate: float):
        super(Linker, self).__init__()
        self.dim = dim
        self.weight = torch.nn.Parameter(-torch.ones(dim) / dim ** 0.5)
        self.dropout = Dropout(dropout_rate)

    def forward(self, reprs: list[Tensor],
                indices: list[Tensor],
                num_iters: int,
                training: bool = True,
                tau: float = 1.) -> list[Tensor]:
        return self.gather_and_link(reprs, indices, num_iters, training, tau)

    def compute_link_strengths(self, negative: Tensor, positive: Tensor) -> Tensor:
        weight = self.dropout(self.weight)
        return contract('bix,bjx,x->bij', negative, positive, weight)  # type: ignore

    def gather_and_link(self,
                        reprs: list[Tensor],
                        indices: list[Tensor],
                        num_iters: int,
                        training: bool = True,
                        tau: float = 1.) -> list[Tensor]:
        negatives, positives = gather_many_leaf_reprs(reprs, indices, training)
        scores = [self.compute_link_strengths(negative, positive) for negative, positive in zip(negatives, positives)]
        return [sinkhorn(score, tau, num_iters) for score in scores]


def norm(x: Tensor, dim: int) -> Tensor:
    return x - logsumexp(x, dim=dim, keepdim=True)


def step(x: Tensor) -> Tensor:
    return norm(norm(x, -1), -2)


def sinkhorn(x: Tensor, tau: float, num_iters: int) -> Tensor:
    x = x/tau
    for _ in range(num_iters):
        x = step(x)
    return x


def gather_leaf_reprs(reprs: list[Tensor],
                      indices: Tensor,
                      skip_uneven_batches: bool = True) -> tuple[Tensor, Tensor] | None:
    if skip_uneven_batches:
        max_depth = len(reprs) - 1
        max_depth_mask = (indices[..., 0].max(dim=-1).values <= max_depth).prod(-1).bool()
        if not max_depth_mask.any():
            return None
        indices = indices[max_depth_mask]
    sizes = indices.shape[:-1]
    indices = indices.flatten(0, -2)
    ret = torch.stack([reprs[level][index] for level, index in indices]).unflatten(0, sizes)
    negative, positive = ret.chunk(2, dim=-2)
    assert negative.shape == positive.shape
    return negative.squeeze(-2), positive.squeeze(-2)


def gather_many_leaf_reprs(reprs: list[Tensor],
                           indices: list[Tensor],
                           training: bool) -> tuple[list[Tensor], list[Tensor]]:
    return tuple(zip(*(g for index in indices if (g := gather_leaf_reprs(reprs, index, training)) is not None)))
