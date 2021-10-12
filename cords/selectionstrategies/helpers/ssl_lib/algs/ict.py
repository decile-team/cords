import torch
from .consistency import ConsistencyRegularization
from .utils import mixup

class ICT(ConsistencyRegularization):
    """
    Interpolation Consistency Training https://arxiv.org/abs/1903.03825

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
    alpha: float
        beta distribution parameter
    """
    def __init__(
        self,
        consistency,
        threshold: float = 1.,
        sharpen: float = None,
        temp_softmax: float = None,
        alpha: float = 0.1
    ):
        super().__init__(
            consistency,
            threshold,
            sharpen,
            temp_softmax
        )
        self.alpha = alpha

    def __call__(self, tea_logits, w_data, stu_forward, subset=False, *args, **kwargs):
        mask = self.gen_mask(tea_logits)
        targets = self.adjust_target(tea_logits)
        mixed_x, mixed_targets = mixup(w_data, targets, self.alpha)
        if subset:
            y, l1 = stu_forward(mixed_x, last=True, freeze=True)
            return y, l1, mixed_targets, mask
        else:
            y = stu_forward(mixed_x)
            return y, mixed_targets, mask

    def __repr__(self):
        return f"ICT(threshold={self.threshold}, sharpen={self.sharpen}, tau={self.tau}, alpha={self.alpha})"

