from torch.nn import Module, CrossEntropyLoss
from torch.nn.functional import nll_loss
from torch import Tensor, arange


class TaggingLoss(Module):
    def __init__(self, reduction: str, label_smoothing: float = 0.0):
        super(TaggingLoss, self).__init__()
        self._loss_fn = CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, logits: list[Tensor], labels: list[Tensor]) -> Tensor:
        return sum(map(self._loss_fn, logits, labels))


class LinkingLoss(Module):
    def __init__(self):
        super(LinkingLoss, self).__init__()

    def forward_one(self, match: Tensor) -> Tensor:
        batch_size, num_candidates = match.shape[:-1]
        return nll_loss(match.flatten(0, -2),
                        arange(num_candidates).repeat(batch_size).to(match.device),
                        reduction='mean')

    def forward(self, matches: list[Tensor]) -> Tensor:
        return sum(map(self.forward_one, matches))
