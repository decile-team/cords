import torch.nn as nn
import torch.nn.functional as F

def cross_entropy(y, target, mask=None, reduce=True):
    if target.ndim == 1: # for hard label
        loss = F.cross_entropy(y, target, reduction="none")
    else:
        loss = -(target * F.log_softmax(y, 1)).sum(1)
    if mask is not None:
        loss = mask * loss
    if reduce:
        return loss.mean()
    else:
        return loss


class CrossEntropy(nn.Module):

    def __init__(self, reduce):
        super().__init__()
        self.reduce = reduce

    def forward(self, y, target, mask=None, *args, **kwargs):
        return cross_entropy(y, target.detach(), mask, self.reduce)
