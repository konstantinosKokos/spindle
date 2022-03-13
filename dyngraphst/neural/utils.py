import pdb

import torch

from torch import Tensor, sigmoid
from torch.nn import Module, Linear, Parameter
from math import cos, radians

from typing import Callable
from warnings import warn


def make_schedule(warmup_steps: int,
                  total_steps: int,
                  warmdown_steps: int,
                  max_lr: float,
                  min_lr: float) -> Callable[[int], float]:
    linear_schedule = make_linear_schedule(warmup_steps, max_lr)
    cosine_schedule = make_cosine_schedule(warmdown_steps, max_lr, min_lr)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return linear_schedule(step)
        elif step < total_steps - warmdown_steps:
            return max_lr
        elif step > total_steps:
            warn(f"Step is greater than total steps")
            return min_lr
        return cosine_schedule(step - (total_steps - warmdown_steps))
    return schedule


def make_linear_schedule(warmup_steps: int, max_lr: float) -> Callable[[int], float]:
    def linear_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps * max_lr
        return max_lr
    return linear_schedule


def make_cosine_schedule(decay_steps: int, max_lr: float, min_lr: float) -> Callable[[int], float]:
    def cosine_schedule(step: int) -> float:
        if step <= decay_steps:
            return min_lr + (max_lr - min_lr) * (cos(radians(step / decay_steps * 180)) + 1) / 2
        return min_lr
    return cosine_schedule


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


def count_params(model: Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)