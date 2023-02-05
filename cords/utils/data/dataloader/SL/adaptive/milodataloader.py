from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import WeightedRandomExplorationStrategy
from torch.utils.data import DataLoader
from .weightedrandomdataloader import WeightedRandomDataLoader
from .stochasticgreedydataloader import StochasticGreedyDataLoader
import time, math


class MILODataLoader(AdaptiveDSSDataLoader):
    """
    Implements of MILODataLoader that serves as the dataloader for the adaptive CRAIG subset selection strategy from the paper :footcite:`pmlr-v119-mirzasoleiman20a`.

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
        self.strategy = WeightedRandomExplorationStrategy(train_loader, 
                                            dss_args.global_order_file, 
                                            online=True, 
                                            temperature=dss_args.temperature, 
                                            per_class=dss_args.per_class)
        super(MILODataLoader, self).__init__(train_loader, train_loader, dss_args, 
                                                 logger, *args, **kwargs)
        """
         Arguments assertion check
        """
        assert "global_order_file" in dss_args.keys(), "'global_order_file' is a compulsory argument. Include it as a key in dss_args"
        #assert "facloc_stochastic_subsets_file" in dss_args.keys(), "'facloc_stochastic_subsets_file' is a compulsory argument. Include it as a key in dss_args"
        assert "gc_stochastic_subsets_file" in dss_args.keys(), "'gc_stochastic_subsets_file' is a compulsory argument. Include it as a key in dss_args"
        assert "temperature" in dss_args.keys(), "'temperature' is a compulsory argument. Include it as a key in dss_args"
        assert "per_class" in dss_args.keys(), "'per_class' is a compulsory argument. Include it as a key in dss_args"
        assert "num_epochs" in dss_args.keys(), "'num_epochs' is a compulsory argument when warm starting the model(i.e., kappa > 0). Include it as a key in dss_args"
        assert "gc_ratio" in dss_args.keys(), "'gc_ratio' is a compulsory argument when warm starting the model(i.e., kappa > 0). Include it as a key in dss_args"
        
        self.logger.debug('Hybrid Global order dataloader initialized.')
        self.num_epochs = dss_args.num_epochs
        self.olgo_loader = WeightedRandomDataLoader(train_loader, dss_args, logger, *args, **kwargs)
        self.gc_ratio = dss_args.gc_ratio
        dss_args.stochastic_subsets_file = dss_args.gc_stochastic_subsets_file
        self.gc_stochasticgo_loader = StochasticGreedyDataLoader(train_loader, dss_args, logger, *args, **kwargs)

    def __iter__(self):
        """
        Iter function that returns the curriculum of easy to hard subsets using stochastic greedy exploration and weighted random exploration based on the value of gc_ratio.
        """
        self.initialized = True
        if self.cur_epoch < math.ceil(self.gc_ratio * self.num_epochs):
            self.logger.debug(
                "Using GC Stochastic Data Loader from epoch {0:d}".format(self.cur_epoch, math.ceil((1/12) * self.num_epochs)))
            loader = self.gc_stochasticgo_loader
        else:
            self.logger.debug('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
            loader = self.olgo_loader
            self.logger.debug('Epoch: {0:d}, finished reading dataloader. '.format(self.cur_epoch))
        self.cur_epoch += 1
        return loader.__iter__()

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
        start = time.time()
        self.logger.debug('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        subset_indices, subset_weights = self.strategy.select(self.budget)
        end = time.time()
        self.logger.info('Epoch: {0:d}, GlobalOrder based subset selection finished, takes {1:.4f}. '.format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights

    def _resample_subset_indices(self):
        pass