import numpy as np
from . import augmentation_pool
from . import utils


class RandAugment:
    """
    RandAugment class

    Parameters
    --------
    nops: int
        number of operations per image
    magnitude: int
        maximmum magnitude
    alg: str
        algorithm name
    """
    def __init__(self, nops=2, magnitude=10, prob=0.5, alg="fixmatch"):
        self.nops = nops
        self.magnitude = magnitude
        self.prob = prob
        if alg == "fixmatch":
            self.ops_list = utils.FIXMATCH_RANDAUGMENT_OPS_LIST
        elif alg == "uda":
            self.ops_list = utils.UDA_RANDAUGMENT_OPS_LIST
        else:
            raise NotImplementedError

        self.ops_max_level = utils.RANDAUGMENT_MAX_LEVELS

    def __call__(self, img):
        """
        Apply augmentations to PIL image
        """
        ops = np.random.choice(self.ops_list, self.nops)
        for name in ops:
            if np.random.rand() <= self.prob:
                level = np.random.randint(1, self.magnitude)
                max_level = self.ops_max_level[name]
                transform = getattr(augmentation_pool, name)
                img = transform(img, level, magnitude=self.magnitude, max_level=max_level)
        return img

    def __repr__(self):
        return f"RandAugment(nops={self.nops}, magnitude={self.magnitude})"
