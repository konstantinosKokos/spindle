import pdb

from torch.nn import Module, CrossEntropyLoss
from torch import Tensor


class GroupedLoss(Module):
    def __init__(self, reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()
        self.loss_fn = CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
        self.reduction = reduction

    def forward_many(self, predictions: list[Tensor], targets: list[Tensor]) -> Tensor:
        return sum(map(self.forward, predictions, targets))

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        loss = self.loss_fn(prediction, target)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'Reduction {self.reduction} not supported')

