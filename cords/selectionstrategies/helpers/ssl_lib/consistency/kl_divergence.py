import torch.nn as nn
import torch.nn.functional as F
import torch

def kl_divergence(y, target, mask=None, reduce=True):
    loss = ((target * torch.log(target)) - (target * F.log_softmax(y, 1))).sum(1)
    if mask is not None:
        loss = mask * loss
    if reduce:
        return loss.mean()
    else:
        return loss


class KLDivergence(nn.Module):

    def __init__(self, reduce):
        super().__init__()
        self.reduce = reduce

    def forward(self, y, target, mask=None, *args, **kwargs):
        return kl_divergence(y, target.detach(), mask, self.reduce)
