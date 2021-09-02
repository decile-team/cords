from abc import abstractmethod
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from cords.selectionstrategies.supervisedlearning import GLISTERStrategy, RandomStrategy
import copy, logging, time
import numpy as np


class DSSDataLoader:
    def __init__(self, full_data, budget, verbose=False, *args, **kwargs):
        super(DSSDataLoader, self).__init__()
        # TODO: Integrate verbose in logging
        self.verbose = verbose
        # self.full_data = full_data
        self.dataset = full_data
        self.budget = budget
        self.subset_loader = None
        # Subset
        self.strategy = None
        self.loader_args = args
        self.loader_kwargs = kwargs
        self._init()

    def __getattr__(self, item):
        self.subset_loader.__getattribute__(item)

    def resample(self):
        subset_indices = self._resample_subset_indices()
        logging.debug("Subset indices length: %d", len(subset_indices))
        self._refresh_subset_loader(subset_indices)
        logging.debug("Subset loader inited, args: %s, kwargs: %s", self.loader_args, self.loader_kwargs)
        logging.info('Sample finished, total number of data: %d, number of subset: %d', len(self.dataset), len(self.subset_loader.dataset))

    def _init(self):
        random_indices = np.random.choice(len(self.dataset), size=self.budget, replace=False)
        self._refresh_subset_loader(random_indices)

    def _refresh_subset_loader(self, indices):
        self.subset_loader = DataLoader(Subset(self.dataset, indices), *self.loader_args, **self.loader_kwargs)

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self):
        pass

