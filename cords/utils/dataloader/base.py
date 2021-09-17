from abc import abstractmethod
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
import logging
import numpy as np


# Base objects

class DSSDataLoader:
    def __init__(self, full_data, budget, verbose=False, *args, **kwargs):
        super(DSSDataLoader, self).__init__()
        # TODO: Integrate verbose in logging
        self.len_full = len(full_data)
        assert budget < self.len_full, "Budget should smaller than full length. Budget: %s / Full length: %s" % (
            budget, self.len_full)
        self.verbose = verbose
        self.dataset = full_data
        self.budget = budget
        self.loader_args = args
        self.loader_kwargs = kwargs
        self.fullset_loader = DataLoader(self.dataset, *self.loader_args, **self.loader_kwargs)
        self.subset_indices = None
        self.subset_loader = None
        self.batch_wise_indices = None
        self.strategy = None
        self.cur_epoch = 0
        self._init_subset_loader()

    def __getattr__(self, item):
        return object.__getattribute__(self, "subset_loader").__getattribute__(item)

    def resample(self):
        self.subset_indices = self._resample_subset_indices()
        logging.debug("Subset indices length: %d", len(self.subset_indices))
        self._refresh_subset_loader(self.subset_indices)
        logging.debug("Subset loader inited, args: %s, kwargs: %s", self.loader_args, self.loader_kwargs)
        logging.info('Sample finished, total number of dataloader: %d, number of subset: %d',
                     self.len_full, len(self.subset_loader.dataset))

    def _init_subset_loader(self):
        # All strategies start with random selection
        self.subset_indices = self._init_subset_indices()
        self._refresh_subset_loader(self.subset_indices)

    # Default subset indices comes from random selection
    def _init_subset_indices(self):
        return np.random.choice(len(self.dataset), size=self.budget, replace=False)

    def _refresh_subset_loader(self, indices):
        # data_sub = Subset(trainset, idxs)
        # subset_trnloader = torch.utils.data.DataLoader(data_sub, batch_size=trn_batch_size, shuffle=False,
        #                                                pin_memory=True)
        self.subset_loader = DataLoader(Subset(self.dataset, indices), shuffle=False,
                                        *self.loader_args, **self.loader_kwargs)
        self.batch_wise_indices = list(self.subset_loader.batch_sampler)

    # TODO: checkpoints
    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self):
        pass
