import pdb

import torch
from torch import Tensor, logsumexp
from torch.nn import Module
from opt_einsum import contract


class Linker(Module):
    def __init__(self, dim: int):
        super(Linker, self).__init__()
        self.weight = torch.nn.Parameter(torch.eye(dim, dim))

    def forward(self, reprs: list[Tensor],
                indices: list[Tensor],
                num_iters: int,
                train: bool = True,
                tau: int = 1) -> list[Tensor]:
        return self.gather_and_link(reprs, indices, num_iters, train, tau)

    def compute_link_strengths(self, negative: Tensor, positive: Tensor) -> Tensor:
        return contract('bix,bjy,xy->bij', negative, positive, self.weight)  # type: ignore

    def gather_and_link(self,
                        reprs: list[Tensor],
                        indices: list[Tensor],
                        num_iters: int,
                        train: bool = True,
                        tau: int = 1) -> list[Tensor]:
        negatives, positives = gather_many_leaf_reprs(reprs, indices, train)
        scores = [self.compute_link_strengths(negative, positive) for negative, positive in zip(negatives, positives)]
        return [sinkhorn(score, tau, num_iters) for score in scores]


def norm(x: Tensor, dim: int) -> Tensor:
    return x - logsumexp(x, dim=dim, keepdim=True)


def step(x: Tensor) -> Tensor:
    return norm(norm(x, -1), -2)


def sinkhorn(x: Tensor, tau: int, num_iters: int) -> Tensor:
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
                           train: bool) -> tuple[list[Tensor], list[Tensor]]:
    return tuple(zip(*(g for index in indices if (g := gather_leaf_reprs(reprs, index, not train)) is not None)))
