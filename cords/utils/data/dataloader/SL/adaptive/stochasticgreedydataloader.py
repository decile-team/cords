from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import StochasticGreedyExplorationStrategy
import time


class StochasticGreedyDataLoader(AdaptiveDSSDataLoader):
    """
    Implements of StochasticGreedyDataLoader that serves as the dataloader for the adaptive CRAIG subset selection strategy from the paper :footcite:`pmlr-v119-mirzasoleiman20a`.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    dss_args: dict
        Data subset selection arguments dictionary required for CRAIG subset selection strategy
    logger: class
        Logger for logging the information
    """
    def __init__(self, train_loader, dss_args, logger, *args, **kwargs):
        self.strategy = StochasticGreedyExplorationStrategy(train_loader, 
                                            dss_args.stochastic_subsets_file, 
                                            )
        super(StochasticGreedyDataLoader, self).__init__(train_loader, train_loader, dss_args, 
                                                 logger, *args, **kwargs)
        """
         Arguments assertion check
        """
        assert "stochastic_subsets_file" in dss_args.keys(), "'stochastic_subsets_file' is a compulsory argument. Include it as a key in dss_args"
        self.logger.debug('Global order stochastic dataloader initialized.')

    # Over-riding initial random subset selection
    def _init_subset_loader(self):
        """
        Function that initializes the initial subset loader
        """
        self.subset_indices, self.subset_weights = self._init_subset_indices()
        self._refresh_subset_loader()

    def _init_subset_indices(self):
        """
        Function that initializes the initial subset indices
        """
        self.logger.debug('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        start = time.time()
        subset_indices, subset_weights = self.strategy.select(self.budget)
        end = time.time()
        self.logger.info('Epoch: {0:d}, GlobalOrder stochastic subset selection finished, takes {1:.4f}. '.format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights

    def _resample_subset_indices(self):
        """
        Function that calls the Stochastic Greedy subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        print("Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch))
        self.logger.debug("Stochastic GlobalOrder budget: %d", self.budget)
        start = time.time()
        subset_indices, subset_weights = self.strategy.select(self.budget)
        end = time.time()
        self.logger.info("Epoch: {0:d}, GlobalOrder stochastic subset selection finished, takes {1:.4f}. ".format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights
