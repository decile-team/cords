from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import RandomStrategy
import time, copy, logging

class RandomDataLoader(AdaptiveDSSDataLoader):
    def __init__(self, train_loader, dss_args, verbose=True, *args, **kwargs):
        super(RandomDataLoader, self).__init__(train_loader, train_loader, dss_args, 
                                                    verbose=verbose, *args, **kwargs)
        self.strategy = RandomStrategy(train_loader, online=False)
        if self.verbose:
            print('Random dataloader initialized. ')

    def _resample_subset_indices(self):
        if self.verbose:
            start = time.time()
            print("Iteration: {0:d}, requires subset selection. ".format(self.cur_iter))
        logging.debug("Random budget: %d", self.budget)
        subset_indices, _ = self.strategy.select(self.budget)
        if self.verbose:
            end = time.time()
            print("Iteration: {0:d}, subset selection finished, takes {1:.2f}. ".format(self.cur_iter, (end - start)))
        return subset_indices
