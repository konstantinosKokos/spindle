import pdb

from torch.nn import Module, CrossEntropyLoss, NLLLoss
from torch import Tensor, arange


class TaggingLoss(Module):
    def __init__(self, reduction: str = 'mean', label_smoothing: float = 0.0):
        super(TaggingLoss, self).__init__()
        self._loss_fn = CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, logits: list[Tensor], labels: list[Tensor]) -> Tensor:
        return sum(map(self._loss_fn, logits, labels))


class LinkingLoss(Module):
    def __init__(self):
        super(LinkingLoss, self).__init__()
        self._loss_fn = NLLLoss(reduction='mean')

    def forward_one(self, match: Tensor) -> Tensor:
        batch_size, num_candidates = match.shape[:-1]
        return self._loss_fn(match.view(-1, num_candidates),
                             arange(num_candidates).repeat(batch_size).to(match.device))

    def forward(self, matches: list[Tensor]) -> Tensor:
        return sum(map(self.forward_one, matches))


class LossScaler:
    def __init__(self, n: int):
        self.emas: list[float]
        self.alpha: float = 2 / (n + 1)
        self.emvar: list[float]
        self.step = self._first_step

    def _first_step(self, *xs: float):
        self.emas = list(xs)
        self.emvar = [0] * len(xs)
        self.step = self._step
        self.step(*xs)

    def _step(self, *xs):
        for i, x in enumerate(xs):
            delta = x - self.emas[i]
            self.emas[i] = self.emas[i] + self.alpha * delta
            self.emvar[i] = (1 - self.alpha) * (self.emvar[i] + self.alpha * delta**2)

    # class SinkhornLoss(Module):
    #     def __init__(self):
    #         super(SinkhornLoss, self).__init__()
    #
    #     def forward(self, predictions: list[Tensor], truths: list[Tensor]):
    #         return sum(nll_loss(link.flatten(0, 1), perm.flatten(), reduction='mean')
    #                    for link, perm in zip(predictions, truths))



    # def __init__(self, reduction: str = 'mean', label_smoothing: float = 0.0):
    #     super().__init__()
    #     self.loss_fn = GroupedLoss(reduction='none', label_smoothing=label_smoothing)
    #     self.reduction = reduction
    # 
    # def forward(self, predictions: list[Tensor], targets: list[Tensor]) -> Tensor:
    #     return self.loss_fn(predictions, targets)
    # 
    # def forward_many(self, predictions: list[Tensor], targets: list[Tensor]) -> Tensor:
    #     return self.loss_fn.forward_many(predictions, targets)