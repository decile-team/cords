from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import RandomStrategy
import time, copy, logging

class OLRandomDataLoader(AdaptiveDSSDataLoader):
    def __init__(self, train_loader, dss_args, logger, *args, **kwargs):
        super(OLRandomDataLoader, self).__init__(train_loader, train_loader, dss_args, 
                                                logger, *args, **kwargs)
        self.strategy = RandomStrategy(train_loader, online=True)
        self.logger.debug('OLRandom dataloader initialized.')

    def _resample_subset_indices(self):
        start = time.time()
        self.logger.debug("Iteration: {0:d}, requires subset selection. ".format(self.cur_iter))
        logging.debug("Random budget: %d", self.budget)
        subset_indices, _ = self.strategy.select(self.budget)
        end = time.time()
        self.logger.info("Iteration: {0:d}, subset selection finished, takes {1:.2f}. ".format(self.cur_iter, (end - start)))
        return subset_indices
