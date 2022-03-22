from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SSL import CRAIGStrategy
import time, copy


# CRAIG
class CRAIGDataLoader(AdaptiveDSSDataLoader):
    """
    Implements of CRAIGDataLoader that serves as the dataloader for the adaptive CRAIG subset selection strategy for semi-supervised learning
    and is an adapted version from the paper :footcite:`pmlr-v119-mirzasoleiman20a`.

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
        assert "tea_model" in dss_args.keys(), "'tea_model' is a compulsory argument. Include it as a key in dss_args"
        assert "ssl_alg" in dss_args.keys(), "'ssl_alg' is a compulsory argument. Include it as a key in dss_args"
        assert "loss" in dss_args.keys(), "'loss' is a compulsory argument. Include it as a key in dss_args"
        if dss_args.loss.reduce:
            raise ValueError("Please set 'reduce' of loss function to False for adaptive subset selection strategies")
        assert "num_classes" in dss_args.keys(), "'num_classes' is a compulsory argument for CRAIG. Include it as a key in dss_args"
        assert "linear_layer" in dss_args.keys(), "'linear_layer' is a compulsory argument for CRAIG. Include it as a key in dss_args"
        assert "selection_type" in dss_args.keys(), "'selection_type' is a compulsory argument for CRAIG. Include it as a key in dss_args"
        assert "optimizer" in dss_args.keys(), "'optimizer' is a compulsory argument for CRAIG. Include it as a key in dss_args"
        
        super(CRAIGDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                            logger, *args, **kwargs)
        
        self.strategy = CRAIGStrategy(train_loader, val_loader, copy.deepcopy(dss_args.model), copy.deepcopy(dss_args.tea_model), 
                                     dss_args.ssl_alg, dss_args.loss, dss_args.device, dss_args.num_classes, dss_args.linear_layer,  
                                     True, dss_args.selection_type, logger, dss_args.optimizer)
        self.train_model = dss_args.model
        self.teacher_model = dss_args.tea_model
        self.logger.debug('CRAIG dataloader initialized. ')

    def _resample_subset_indices(self):
        """
        Function that calls the CRAIG subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        self.logger.debug('Iteration: {0:d}, requires subset selection. '.format(self.cur_iter))
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        if self.teacher_model is not None:
            tea_cached_state_dict = copy.deepcopy(self.teacher_model.state_dict())
            tea_clone_dict = copy.deepcopy(self.teacher_model.state_dict())
        else:
            tea_clone_dict = None
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict, tea_clone_dict)
        self.train_model.load_state_dict(cached_state_dict)
        if self.teacher_model is not None:
            self.teacher_model.load_state_dict(tea_cached_state_dict)
        end = time.time()
        self.logger.info('Iteration: {0:d}, subset selection finished, takes {1:.2f}. '.format(self.cur_iter, (end - start)))
        return subset_indices, subset_weights
