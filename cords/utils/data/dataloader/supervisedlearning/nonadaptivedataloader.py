import copy
from abc import ABC
import numpy as np
import apricot
import math
import time
from cords.selectionstrategies.supervisedlearning import CRAIGStrategy
from .dssdataloader import DSSDataLoader


class NonAdaptiveDSSDataLoader(DSSDataLoader):
    def __init__(self, train_loader, val_loader, budget, model, loss, device, verbose=False, *args,
                 **kwargs):
        super(NonAdaptiveDSSDataLoader, self).__init__(train_loader.dataset, budget,
                                                       verbose=verbose, *args, **kwargs)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss = copy.deepcopy(loss)
        self.device = device
        self.initialized = False

    def __iter__(self):
        return self.subset_loader.__iter__()
