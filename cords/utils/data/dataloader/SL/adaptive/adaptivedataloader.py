import logging
from abc import abstractmethod
from torch.utils.data import DataLoader
from ..dssdataloader import DSSDataLoader
from math import ceil


class AdaptiveDSSDataLoader(DSSDataLoader):
    """
    Implementation of AdaptiveDSSDataLoader class which serves as base class for dataloaders of other
    adaptive subset selection strategies for supervised learning framework.

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
    def __init__(self, train_loader, val_loader, dss_args, logger, *args,
                 **kwargs):
        
        """
        Constructor function
        """
        super(AdaptiveDSSDataLoader, self).__init__(train_loader.dataset, dss_args,
                                                    logger, *args, **kwargs)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Arguments assertion check
        assert "select_every" in dss_args.keys(), "'select_every' is a compulsory argument. Include it as a key in dss_args"
        assert "device" in dss_args.keys(), "'device' is a compulsory argument. Include it as a key in dss_args"
        assert "kappa" in dss_args.keys(), "'kappa' is a compulsory argument. Include it as a key in dss_args"
        self.select_every = dss_args.select_every
        self.device = dss_args.device
        self.kappa = dss_args.kappa
        if dss_args.kappa > 0:
            assert "num_epochs" in dss_args.keys(), "'num_epochs' is a compulsory argument when warm starting the model(i.e., kappa > 0). Include it as a key in dss_args"
            self.select_after =  int(dss_args.kappa * dss_args.num_epochs)
            self.warmup_epochs = ceil(self.select_after * dss_args.fraction)
        else:
            self.select_after = 0
            self.warmup_epochs = 0
        self.initialized = False
        
    
    def __iter__(self):
        """
        Iter function that returns the iterator of full data loader or data subset loader or empty loader based on the 
        warmstart kappa value.
        """
        self.initialized = True
        if self.warmup_epochs < self.cur_epoch <= self.select_after:
            self.logger.debug(
                "Skipping epoch {0:d} due to warm-start option. ".format(self.cur_epoch, self.warmup_epochs))
            loader = DataLoader([])
            
        elif self.cur_epoch <= self.warmup_epochs:
            self.logger.debug('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
            loader = self.wtdataloader
            self.logger.debug('Epoch: {0:d}, finished reading dataloader. '.format(self.cur_epoch))
        else:
            self.logger.debug('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
            if ((self.cur_epoch - 1) % self.select_every == 0) and (self.cur_epoch > 1):
                self.resample()
            loader = self.subset_loader
            self.logger.debug('Epoch: {0:d}, finished reading dataloader. '.format(self.cur_epoch))
            
        self.cur_epoch += 1
        return loader.__iter__()

    def __len__(self) -> int:
        """
        Returns the length of the current data loader
        """
        if self.warmup_epochs < self.cur_epoch <= self.select_after:
            self.logger.debug(
                "Skipping epoch {0:d} due to warm-start option. ".format(self.cur_epoch, self.warmup_epochs))
            loader = DataLoader([])
            return len(loader)

        elif self.cur_epoch <= self.warmup_epochs:
            self.logger.debug('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
            loader = self.wtdataloader
            #self.logger.debug('Epoch: {0:d}, finished reading dataloader. '.format(self.cur_epoch))
            return len(loader)
        else:
            self.logger.debug('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
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
        """
        Abstract function that needs to be implemented in the child classes. 
        Needs implementation of subset selection implemented in child classes.
        """
        raise Exception('Not implemented.')