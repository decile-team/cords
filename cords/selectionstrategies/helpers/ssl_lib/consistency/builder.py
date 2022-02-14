from .cross_entropy import CrossEntropy
from .mean_squared import MeanSquared
from .kl_divergence import KLDivergence

def gen_consistency(type, cfg):
    # doubts: what is this doing?
    if type == "ce":
        return CrossEntropy(True)
    elif type == "ce_red":
        return CrossEntropy(False)
    elif type == "ms":
        return MeanSquared(True)
    elif type == "ms_red":
        return MeanSquared(False)
    elif type == "kld":
        return KLDivergence(True)
    elif type == "kld_red":
        return KLDivergence(False)
    else:
        return None
