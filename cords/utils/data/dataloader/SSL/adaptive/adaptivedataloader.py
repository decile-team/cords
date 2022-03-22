import logging, torch
from abc import abstractmethod
from torch.utils.data import DataLoader
from ..dssdataloader import DSSDataLoader
from cords.utils.data.datasets.SSL.utils import InfiniteSampler
from cords.utils.data.data_utils import WeightedSubset


class AdaptiveDSSDataLoader(DSSDataLoader):
    """
    Implementation of AdaptiveDSSDataLoader class which serves as base class for dataloaders of other
    adaptive subset selection strategies for semi-supervised learning framework.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary
    logger: class
        Logger for logging the information
    """
    def __init__(self, train_loader, val_loader, dss_args,
                 logger, *args, **kwargs):
        """
        Constructor function
        """
        # Arguments assertion check
        assert "select_every" in dss_args.keys(), "'select_every' is a compulsory argument. Include it as a key in dss_args"
        assert "device" in dss_args.keys(), "'device' is a compulsory argument. Include it as a key in dss_args"
        assert "kappa" in dss_args.keys(), "'kappa' is a compulsory argument. Include it as a key in dss_args"
        assert "num_iters" in dss_args.keys(), "'num_iters' is a compulsory argument. Include it as a key in dss_args"
        assert "batch_size" in kwargs.keys(), "'batch_size' is a compulsory argument. Include it as a key in kwargs"
        assert "sampler" not in kwargs.keys(), "'sampler' is a prohibited argument. Do not include it as a key in kwargs"
        assert "shuffle" not in kwargs.keys(), "'shuffle' is a prohibited argument. Do not include it as a key in kwargs"
        
        self.select_every = dss_args.select_every
        self.sel_iteration = int((self.select_every * len(train_loader.dataset) * dss_args.fraction) // (kwargs['batch_size']))  
        self.device = dss_args.device
        self.kappa = dss_args.kappa
        self.num_iters = dss_args.num_iters
        if dss_args.kappa > 0:
            assert "num_iters" in dss_args.keys(), "'num_iters' is a compulsory argument when warm starting the model(i.e., kappa > 0). Include it as a key in dss_args"
            self.select_after = int(self.kappa * self.num_iters)
        else:
            self.select_after = 0
        super(AdaptiveDSSDataLoader, self).__init__(train_loader.dataset, dss_args,
                                                    logger, *args, **kwargs)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wtdataloader = DataLoader(self.wt_trainset,
                                       sampler=InfiniteSampler(len(self.wt_trainset), self.select_after * kwargs['batch_size']),
                                       *self.loader_args, **self.loader_kwargs)
        self.initialized = False
    
    def _init_subset_loader(self):
        # All strategies start with random selection
        """
        Function that initializes the data subset loader
        """
        self.subset_indices = self._init_subset_indices()
        self.subset_weights = torch.ones(self.budget)
        self._refresh_subset_loader()

    def _refresh_subset_loader(self):
        """
        Function that initializes the subset indices
        """
        data_sub = WeightedSubset(self.dataset, self.subset_indices, self.subset_weights)
        self.subset_loader = DataLoader(data_sub, sampler=InfiniteSampler(len(data_sub), 
                                        self.sel_iteration * self.loader_kwargs['batch_size']),
                                         *self.loader_args, **self.loader_kwargs)
        self.batch_wise_indices = list(self.subset_loader.batch_sampler)
        if self.kappa > 0:
            self.curr_loader = DataLoader(self.wt_trainset, sampler=InfiniteSampler(len(self.wt_trainset), 
                                        self.select_after * self.loader_kwargs['batch_size']),
                                       *self.loader_args, **self.loader_kwargs)
        else:
            self.curr_loader = self.subset_loader


    def __iter__(self):
        """
        Iter function that returns the iterator of full data loader or data subset loader or empty loader based on the 
        warmstart kappa value.
        """
        self.initialized = True
        if self.cur_iter <= self.select_after:
            self.logger.debug('Iteration: {0:d}, reading full dataloader... '.format(self.cur_iter))
            self.curr_loader = self.wtdataloader
            self.logger.debug('Iteration: {0:d}, finished reading full dataloader. '.format(self.cur_iter))
        else:
            self.logger.debug('Iteration: {0:d}, reading subset dataloader... '.format(self.cur_iter))
            if self.cur_iter > 1:
                self.resample()
            self.curr_loader = self.subset_loader
            self.logger.debug('Iteration: {0:d}, finished reading dataloader. '.format(self.cur_iter))
        self.cur_iter += len(list(self.curr_loader.batch_sampler))
        return self.curr_loader.__iter__()

    def __len__(self) -> int:
        """
        Returns the length of the current data loader
        """
        if self.cur_iter <= self.select_after:
            self.logger.debug('Iteration: {0:d}, reading full dataloader... '.format(self.cur_iter))
            loader = self.wtdataloader
            #self.logger.debug('Epoch: {0:d}, finished reading dataloader. '.format(self.cur_epoch))
            return len(loader)
        else:
            self.logger.debug('Iteration: {0:d}, reading subset dataloader... '.format(self.cur_iter))
            loader = self.subset_loader
            return len(loader)
            
    def resample(self):
        """
        Function that resamples the subset indices and recalculates the subset weights
        """
        self.subset_indices, self.subset_weights = self._resample_subset_indices()
        self.logger.debug("Subset indices length: %d", len(self.subset_indices))
        self._refresh_subset_loader()
        self.logger.debug("Subset loader initiated, args: %s, kwargs: %s", self.loader_args, self.loader_kwargs)
        self.logger.debug('Subset selection finished, Training data size: %d, Subset size: %d',
                     self.len_full, len(self.subset_loader.dataset))

    @abstractmethod
    def _resample_subset_indices(self):
        raise Exception('Not implemented. ')