from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import RandomStrategy
import time


class OLRandomDataLoader(AdaptiveDSSDataLoader):
    def __init__(self, train_loader, dss_args, logger, *args, **kwargs):
        super(OLRandomDataLoader, self).__init__(train_loader, train_loader, dss_args, 
                                                 logger, *args, **kwargs)
        self.strategy = RandomStrategy(train_loader, online=True)
        self.logger.debug('OLRandom dataloader initialized.')

    def _resample_subset_indices(self):
        start = time.time()
        print("Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch))
        self.logger.debug("Random budget: %d", self.budget)
        subset_indices, subset_wts = self.strategy.select(self.budget)
        end = time.time()
        self.logger.info("Epoch: {0:d}, OLRandom subset selection finished, takes {1:.4f}. ".format(self.cur_epoch, (end - start)))
        return subset_indices, subset_wts
