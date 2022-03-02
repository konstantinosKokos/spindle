import pdb

from torch.nn import Module, KLDivLoss, CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch import Tensor
import torch


class GroupedLoss(Module):
    def __init__(self,
                 reduction: str = 'mean'):
        super().__init__()
        self.loss_fn = CrossEntropyLoss(reduction='none')
        self.reduction = reduction

    def forward_many(self, predictions: list[Tensor], targets: list[Tensor], numels: list[Tensor]) -> Tensor:
        return sum(map(self.forward, predictions, targets, numels))

    def forward(self, prediction: Tensor, target: Tensor, numel: Tensor) -> Tensor:
        loss = self.loss_fn(prediction, target) * numel

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'Reduction {self.reduction} not supported')
