from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import GradMatchStrategy
import time, copy, torch


class GradMatchDataLoader(AdaptiveDSSDataLoader):
    """
    Implements of GradMatchDataLoader that serves as the dataloader for the adaptive GradMatch subset selection strategy from the paper 
    :footcite:`pmlr-v139-killamsetty21a`.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary required for GradMatch subset selection strategy
    logger: class
        Logger for logging the information
    """
    def __init__(self, train_loader, val_loader, dss_args, logger, *args, **kwargs):
        
        """
         Constructor function
        """
        # Arguments assertion check
        assert "model" in dss_args.keys(), "'model' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "loss" in dss_args.keys(), "'loss' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        if dss_args.loss.reduction != "none":
            raise ValueError("Please set 'reduction' of loss function to 'none' for adaptive subset selection strategies")
        assert "eta" in dss_args.keys(), "'eta' is a compulsory argument. Include it as a key in dss_args"
        assert "num_classes" in dss_args.keys(), "'num_classes' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "linear_layer" in dss_args.keys(), "'linear_layer' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "selection_type" in dss_args.keys(), "'selection_type' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "valid" in dss_args.keys(), "'valid' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "v1" in dss_args.keys(), "'v1' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "lam" in dss_args.keys(), "'lam' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "eps" in dss_args.keys(), "'eps' is a compulsory argument for GradMatch. Include it as a key in dss_args"

        super(GradMatchDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                                  logger, *args, **kwargs)
        self.strategy = GradMatchStrategy(train_loader, val_loader, copy.deepcopy(dss_args.model), dss_args.loss, dss_args.eta,
                                          dss_args.device, dss_args.num_classes, dss_args.linear_layer, dss_args.selection_type,
                                          logger, dss_args.valid, dss_args.v1, dss_args.lam, dss_args.eps)
        self.train_model = dss_args.model
        self.logger.debug('Grad-match dataloader initialized. ')

    def _resample_subset_indices(self):
        """
        Function that calls the GradMatch subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        self.logger.debug("Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch))
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict)
        self.train_model.load_state_dict(cached_state_dict)
        end = time.time()
        self.logger.info("Epoch: {0:d}, GradMatch subset selection finished, takes {1:.4f}. ".format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights
