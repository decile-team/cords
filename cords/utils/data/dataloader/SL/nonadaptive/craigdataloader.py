from .nonadaptivedataloader import NonAdaptiveDSSDataLoader
from cords.selectionstrategies.SL import CRAIGStrategy
import time, copy


# CRAIG
class CRAIGDataLoader(NonAdaptiveDSSDataLoader):
    """
    Implements of CRAIGDataLoader that serves as the dataloader for the nonadaptive CRAIG subset selection strategy from the paper :footcite:`pmlr-v119-mirzasoleiman20a`.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary required for CRAIG subset selection strategy
    logger: class
        Logger for logging the information
    """
    def __init__(self, train_loader, val_loader, dss_args, logger, *args, **kwargs):
        """
         Constructor function
        """
        # Arguments assertion check
        assert "model" in dss_args.keys(), "'model' is a compulsory argument. Include it as a key in dss_args"
        assert "loss" in dss_args.keys(), "'loss' is a compulsory argument. Include it as a key in dss_args"
        if dss_args.loss.reduction != "none":
            raise ValueError("Please set 'reduction' of loss function to 'none' for adaptive subset selection strategies")
        assert "num_classes" in dss_args.keys(), "'num_classes' is a compulsory argument for CRAIG. Include it as a key in dss_args"
        assert "linear_layer" in dss_args.keys(), "'linear_layer' is a compulsory argument for CRAIG. Include it as a key in dss_args"
        assert "selection_type" in dss_args.keys(), "'selection_type' is a compulsory argument for CRAIG. Include it as a key in dss_args"
        assert "optimizer" in dss_args.keys(), "'optimizer' is a compulsory argument for CRAIG. Include it as a key in dss_args"
        assert "if_convex" in dss_args.keys(), "'if_convex' is a compulsory argument for CRAIG. Include it as a key in dss_args"
        
        self.strategy = CRAIGStrategy(train_loader, val_loader, copy.deepcopy(dss_args.model), dss_args.num_classes, 
                                     dss_args.linear_layer, dss_args.loss, dss_args.device, 
                                     False, dss_args.selection_type, logger, dss_args.optimizer)
        
        super(CRAIGDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                              logger, *args, **kwargs)
        
        
        self.train_model = dss_args.model
        self.eta = dss_args.eta
        self.num_cls = dss_args.num_classes
        self.model = dss_args.model
        self.loss = copy.deepcopy(dss_args.loss)
        self.logger.debug('Non-adaptive CRAIG dataloader loader initialized. ')

    def _init_subset_loader(self):
        """
        Function that initializes the subset loader based on the subset indices and the subset weights.
        """
        # All strategies start with random selection
        self.subset_indices, self.subset_weights = self._init_subset_indices()
        self._refresh_subset_loader()

    def _init_subset_indices(self):
        """
        Function that calls the CRAIG strategy for initial subset selection and calculating the initial subset weights.
        """
        start = time.time()
        self.logger.debug('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict)
        self.train_model.load_state_dict(cached_state_dict)
        end = time.time()
        self.logger.info('Epoch: {0:d}, CRAIG subset selection finished, takes {1:.4f}. '.format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights
