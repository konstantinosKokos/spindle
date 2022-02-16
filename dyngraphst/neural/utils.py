import pdb

import torch

from torch import Tensor, sigmoid
from torch.nn import Module, Linear, Parameter

from typing import Callable


def make_linear_schedule(warmup_steps: int,
                         warmdown_steps: int,
                         total_steps: int,
                         max_lr: float,
                         min_lr: float) -> Callable[[int], float]:
    def linear_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps * max_lr
        elif step > total_steps - warmdown_steps:
            return (total_steps - step) / warmdown_steps * (max_lr - min_lr) + min_lr
        else:
            return max_lr
    return linear_schedule


def swish(x: Tensor, b: int = 1) -> Tensor:
    return x * sigmoid(b * x)


class SwiGLU(Module):
    def __init__(self, input_dim: int, interm_dim: int):
        super(SwiGLU, self).__init__()
        self.w_in = Linear(input_dim, interm_dim)
        self.v = Linear(input_dim, interm_dim)
        self.w_out = Linear(interm_dim, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        interm = self.w_in(x)
        interm = swish(interm) * self.v(x)
        return self.w_out(interm)


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g