import torch
from .consistency import ConsistencyRegularization

class VAT(ConsistencyRegularization):
    """
    Virtual Adversarial Training https://arxiv.org/abs/1704.03976

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
    objective: function
        objective function
    eps: float
        virtual adversarial noise norm
    xi: float
        perturbation for finite differential method
    n_iter: int
        number of iterations for power method
    """
    def __init__(
        self,
        consistency,
        threshold: float = 1.,
        sharpen: float = None,
        temp_softmax: float = None,
        objective = None,
        eps = 1.0,
        xi = 1e-6,
        n_iter = 1
    ):
        super().__init__(
            consistency,
            threshold,
            sharpen,
            temp_softmax
        )
        self.eps = eps
        self.xi  = xi
        self.n_iter = n_iter
        self.obj_func = objective

    def __call__(self, tea_logits, stu_forward, w_data, subset=False, *args, **kwargs):
        mask = self.gen_mask(tea_logits)
        targets = self.adjust_target(tea_logits)
        d = torch.randn_like(w_data)
        d = self.__normalize(d)
        for _ in range(self.n_iter):
            d.requires_grad = True
            x_hat = w_data + self.xi * d
            y = stu_forward(x_hat)
            loss = self.obj_func(y, targets)
            d = torch.autograd.grad(loss, d)[0]
            d = self.__normalize(d).detach()
        x_hat = w_data + self.eps * d
        if subset:
            y, l1 = stu_forward(x_hat, last=True, freeze=True)
            return y, l1, targets, mask
        else:
            y = stu_forward(x_hat)
            return y, targets, mask

    def __normalize(self, v):
        v = v / (1e-12 + self.__reduce_max(v.abs(), range(1, len(v.shape)))) # to avoid overflow by v.pow(2)
        v = v / (1e-6 + v.pow(2).sum(list(range(1, len(v.shape))), keepdim=True)).sqrt()
        return v

    def __reduce_max(self, v, idx_list):
        for i in idx_list:
            v = v.max(i, keepdim=True)[0]
        return v

    def __repr__(self):
        return f"VAT(threshold={self.threshold}, \
            sharpen={self.sharpen}, \
            tau={self.tau}, \
            eps={self.eps}), \
            xi={self.xi}"
