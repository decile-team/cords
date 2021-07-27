import torch
import warnings
import math
import torch.optim as optim


def exp_warmup(base_value, max_warmup_iter, cur_step):
    """exponential warmup proposed in mean teacher

    calcurate
    base_value * exp(-5(1 - t)^2), t = cur_step / max_warmup_iter

    Parameters
    -----
    base_value: float
        maximum value
    max_warmup_iter: int
        maximum warmup iteration
    cur_step: int
        current iteration
    """
    if max_warmup_iter <= cur_step:
        return base_value
    return base_value * math.exp(-5 * (1 - cur_step/max_warmup_iter)**2)


def linear_warmup(base_value, max_warmup_iter, cur_step):
    """linear warmup

    calcurate
    base_value * (cur_step / max_warmup_iter)
    
    Parameters
    -----
    base_value: float
        maximum value
    max_warmup_iter: int
        maximum warmup iteration
    cur_step: int
        current iteration
    """
    if max_warmup_iter <= cur_step:
        return base_value
    return base_value * cur_step / max_warmup_iter


def cosine_decay(base_lr, max_iteration, cur_step):
    """cosine learning rate decay
    
    cosine learning rate decay with parameters proposed FixMatch
    base_lr * cos( (7\pi cur_step) / (16 max_warmup_iter) )

    Parameters
    -----
    base_lr: float
        maximum learning rate
    max_warmup_iter: int
        maximum warmup iteration
    cur_step: int
        current iteration
    """
    return base_lr * (math.cos( (7*math.pi*cur_step) / (16*max_iteration) ))


def CosineAnnealingLR(optimizer, max_iteration):
    """
    generate cosine annealing learning rate scheduler as LambdaLR
    """
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda cur_step : math.cos((7*math.pi*(cur_step % max_iteration)) / (16*max_iteration)))
