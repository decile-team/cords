import torch
from .utils import sharpening, tempereture_softmax

class ConsistencyRegularization:
    """
    Basis Consistency Regularization

    Parameters
    --------
    consistency: str
        consistency objective name
    threshold: float
        threshold to make mask
    sharpen: float
        sharpening temperature for target value
    temp_softmax: float
        temperature for temperature softmax
    """
    def __init__(
        self,
        consistency,
        threshold: float = None,
        sharpen: float = None,
        temp_softmax: float = None
    ):
        self.consistency = consistency
        self.threshold = threshold
        self.sharpen = sharpen
        self.tau = temp_softmax

    def __call__(
        self,
        stu_preds,
        tea_logits,
        *args,
        **kwargs
    ):
        mask = self.gen_mask(tea_logits)
        targets = self.adjust_target(tea_logits)
        return stu_preds, targets, mask

    def adjust_target(self, targets):
        if self.sharpen is not None:
            targets = targets.softmax(1)
            targets = sharpening(targets, self.sharpen)
        elif self.tau is not None:
            targets = tempereture_softmax(targets, self.tau)
        else:
            targets = targets.softmax(1)
        return targets

    def gen_mask(self, targets):
        targets = targets.softmax(1)
        if self.threshold is None or self.threshold == 0:
            return torch.ones_like(targets.max(1)[0])
        return (targets.max(1)[0] >= self.threshold).float()

    def __repr__(self):
        return f"Consistency(threshold={self.threshold}, sharpen={self.sharpen}, tau={self.tau})"
