from abc import abstractmethod
from .dssdataloader import DSSDataLoader
import logging


class AdaptiveDSSDataLoader(DSSDataLoader):
    def __init__(self, train_loader, val_loader, select_ratio, select_every, model, loss, device, verbose=False, *args,
                 **kwargs):
        super(AdaptiveDSSDataLoader, self).__init__(train_loader.dataset, int(select_ratio*len(train_loader.dataset)),
                                                  verbose=verbose, *args, **kwargs)
        self.cur_epoch = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.select_every = select_every
        self.model = model
        self.loss = loss
        self.device = device

    def __iter__(self):
        if self.verbose:
            logging.info('Epoch: {0:d}, reading data... '.format(self.cur_epoch))
        if self.cur_epoch > 0 and self.cur_epoch % self.select_every == 0:
        # if self.cur_epoch % self.select_every == 0:
            self.resample()
        if self.verbose:
            logging.info('Epoch: {0:d}, finished reading data. '.format(self.cur_epoch))
        self.cur_epoch += 1
        return self.subset_loader.__iter__()

    @abstractmethod
    def _resample_subset_indices(self):
        raise Exception('Not implemented. ')

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass


