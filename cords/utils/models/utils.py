import math
import torch.nn as nn
import torch.nn.functional as F
import torch


def apply_weight_decay(modules, decay_rate):
    """apply weight decay to weight parameters in nn.Conv2d and nn.Linear"""
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data -= decay_rate * m.weight.data


def param_init(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            f, _, k, _ = m.weight.shape
            nn.init.normal_(m.weight, 0, 1./math.sqrt(0.5 * k * k * f))
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


"""
For exponential moving average
"""
def __ema(p1, p2, factor):
    return factor * p1 + (1 - factor) * p2


def __param_update(ema_model, raw_model, factor):
    """ema for trainable parameters"""
    for ema_p, raw_p in zip(ema_model.parameters(), raw_model.parameters()):
        ema_p.data = __ema(ema_p.data, raw_p.data, factor)


def __buffer_update(ema_model, raw_model, factor):
    """ema for buffer parameters (e.g., running_mean and running_var in nn.BatchNorm2d)"""
    for ema_p, raw_p in zip(ema_model.buffers(), raw_model.buffers()):
        ema_p.data = __ema(ema_p.data, raw_p.data, factor)
    # """copy buffer parameters (e.g., running_mean and running_var in nn.BatchNorm2d)"""
    # for ema_p, raw_p in zip(ema_model.buffers(), raw_model.buffers()):
    #     ema_p.copy_(raw_p)


def ema_update(ema_model, raw_model, ema_factor, weight_decay_factor=None, global_step=None):
    if global_step is not None:
        ema_factor = min(1 - 1 / (global_step+1), ema_factor)
    __param_update(ema_model, raw_model, ema_factor)
    __buffer_update(ema_model, raw_model, ema_factor)
    if weight_decay_factor is not None:
        apply_weight_decay(ema_model.modules(), weight_decay_factor)

