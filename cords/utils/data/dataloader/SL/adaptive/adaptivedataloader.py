import logging
from abc import abstractmethod
from torch.utils.data import DataLoader
from ..dssdataloader import DSSDataLoader


class AdaptiveDSSDataLoader(DSSDataLoader):
    def __init__(self, train_loader, val_loader, dss_args, verbose=False, *args,
                 **kwargs):
        super(AdaptiveDSSDataLoader, self).__init__(train_loader.dataset, dss_args,
                                                    verbose=verbose, *args, **kwargs)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        """
         Arguments assertion check
        """
        assert "select_every" in dss_args.keys(), "'select_every' is a compulsory argument. Include it as a key in dss_args"
        assert "device" in dss_args.keys(), "'device' is a compulsory argument. Include it as a key in dss_args"
        assert "kappa" in dss_args.keys(), "'kappa' is a compulsory argument. Include it as a key in dss_args"
        self.select_every = dss_args.select_every
        self.device = dss_args.device
        self.kappa = dss_args.kappa
        if dss_args.kappa > 0:
            assert "num_epochs" in dss_args.keys(), "'num_epochs' is a compulsory argument when warm starting the model(i.e., kappa > 0). Include it as a key in dss_args"
            self.select_after =  int(dss_args.kappa * dss_args.num_epochs)
            self.warmup_epochs = round(self.select_after * dss_args.fraction)
        else:
            self.select_after = 0
            self.warmup_epochs = 0
        self.initialized = False
        
    
    def __iter__(self):
        self.initialized = True
        if self.warmup_epochs < self.cur_epoch <= self.select_after:
            logging.info(
                "Skipping epoch {0:d} due to warm-start option. ".format(self.cur_epoch, self.warmup_epochs))
            loader = DataLoader([])
        else:
            if self.verbose:
                logging.info('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
            if self.cur_epoch <=  self.warmup_epochs:
                loader = self.wtdataloader
            else:
                if ((self.cur_epoch - 1) % self.select_every == 0) and (self.cur_epoch > 1):
                    self.resample()
                loader = self.subset_loader
            if self.verbose:
                logging.info('Epoch: {0:d}, finished reading dataloader. '.format(self.cur_epoch))
        self.cur_epoch += 1
        return loader.__iter__()

    def resample(self):
        self.subset_indices, self.subset_weights = self._resample_subset_indices()
        logging.debug("Subset indices length: %d", len(self.subset_indices))
        self._refresh_subset_loader()
        logging.debug("Subset loader initiated, args: %s, kwargs: %s", self.loader_args, self.loader_kwargs)
        logging.info('Subset selection finished, Training data size: %d, Subset size: %d',
                     self.len_full, len(self.subset_loader.dataset))

    @abstractmethod
    def _resample_subset_indices(self):
        raise Exception('Not implemented. ')