import pdb

from torch.nn import Module, KLDivLoss, CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch import Tensor
import torch


class GroupedLoss(Module):
    def __init__(self,
                 last_unary_index: int,
                 num_classes: int,
                 alpha: float = 0.,
                 token_type_error_scale: float = 2,
                 reduction: str = 'mean'):
        super().__init__()
        self.group_size = last_unary_index
        self.loss_fn = KLDivLoss(reduction='none') if alpha > 0 else CrossEntropyLoss(reduction='none')
        self.reduction = reduction
        self.alpha = alpha
        self.token_type_error_scale = token_type_error_scale
        if self.alpha:
            self.unary_redisitribution = alpha / (last_unary_index - 2)
            self.binary_redisitribution = alpha / (num_classes - last_unary_index - 1)

    def forward(self, prediction: Tensor, target: Tensor, depth_scale: float = 1) -> Tensor:
        p_type = prediction.argmax(dim=-1) < self.group_size
        t_type = target < self.group_size

        if self.alpha:
            p_type = p_type.unsqueeze(-1)
            t_type = t_type.unsqueeze(-1)
            target = torch.zeros_like(prediction)
            target[:, 1:self.group_size].masked_fill_(t_type, self.unary_redisitribution)
            target[:, self.group_size:].masked_fill_(~t_type, self.binary_redisitribution)
            target.scatter_(1, target.unsqueeze(-1), 1 - self.alpha)
            prediction = log_softmax(prediction, dim=-1)

        loss_scales = torch.ones_like(target) * depth_scale
        loss_scales.masked_fill_(p_type != t_type, self.token_type_error_scale * depth_scale)
        loss = self.loss_fn(prediction, target) * loss_scales

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'Reduction {self.reduction} not supported')

