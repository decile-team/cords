from .nonadaptivedataloader import NonAdaptiveDSSDataLoader
from cords.selectionstrategies.SL import WeightedRandomExplorationStrategy
import time, copy


class MILOFixedDataLoader(NonAdaptiveDSSDataLoader):
    """
    Implements of MILOFixedDataLoader that serves as the dataloader for the adaptive CRAIG subset selection strategy from the paper :footcite:`pmlr-v119-mirzasoleiman20a`.

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
        self.strategy = WeightedRandomExplorationStrategy(train_loader, dss_args.global_order_file, online=False, per_class=dss_args.per_class)
        super(MILOFixedDataLoader, self).__init__(train_loader, train_loader, dss_args, 
                                                 logger, *args, **kwargs)
        
        """
         Arguments assertion check
        """
        assert "global_order_file" in dss_args.keys(), "'global_order_file' is a compulsory argument. Include it as a key in dss_args"
        assert "per_class" in dss_args.keys(), "'per_class' is a compulsory argument. Include it as a key in dss_args"
        
        self.logger.debug('Global order dataloader initialized.')

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
    
    def __len__(self) -> int:
        self.logger.debug('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
        loader = self.subset_loader
        return len(loader)