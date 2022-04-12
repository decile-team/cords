from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import SELCONstrategy
import time, copy


# SELCONstrategy
class SELCONDataLoader(AdaptiveDSSDataLoader):
    """
    Implementation of SELCONdataloader that serves as the dataloader for SELCON algorithm from the
    paper :footcite:`durga2021training`.

    Parameters
    -----------
    trainset : torch.utils.data.Dataset class
        Dataset for training.
    validset : torch.utils.data.Dataset class
        Dataset for validation.
    train_loader : torch.utils.data.DataLoader class
        Training data loader.
    val_loader : torch.utils.data.DataLoader class
        Validation data loader.
    dss_args : dict
        Dictionary of arguments for SELCONdataloader.
    logger : logging.Logger class
        Logger for logging.

    """
    def __init__(self, trainset, validset, train_loader, val_loader, dss_args, logger, *args, **kwargs):
        """
        Constructor function
        """
        assert "model" in dss_args.keys(), "'model' is a compulsory argument. Include it as a key in dss_args"
        assert "loss" in dss_args.keys(), "'loss' is a compulsory argument. Include it as a key in dss_args"
        if dss_args.loss.reduction != "none":
            raise ValueError("Please set 'reduction' of loss function to 'none' for adaptive subset selection strategies")
        assert "device" in dss_args.keys(), "'device' is a compulsory argument. Include it as a key in dss_args"
        assert "num_classes" in dss_args.keys(), "'num_classes' is a compulsory argument for SELCON. Include it as a key in dss_args"
        assert "delta" in dss_args.keys(), "'delta' is a compulsory argument for SELCON. Include it as a key in dss_args"
        assert "linear_layer" in dss_args.keys(), "'linear_layer' is a compulsory argument for SELCON. Include it as a key in dss_args"
        
        super(SELCONDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                                logger, *args, **kwargs)
        
        self.strategy = SELCONstrategy(trainset, validset, train_loader, val_loader, copy.deepcopy(dss_args.model), dss_args.loss, dss_args.device,
                                        dss_args.num_classes, dss_args.delta, dss_args.num_epochs, dss_args.linear_layer, dss_args.lam, dss_args.lr, 
                                        logger, dss_args.optimizer, dss_args.batch_size, dss_args.criterion)

        self.train_model = dss_args.model
        self.logger.debug('SELCON dataloader initialized. ')

    def _resample_subset_indices(self):
        """
        Function that calls the SELCON subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        self.logger.debug('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict)
        self.train_model.load_state_dict(cached_state_dict)
        end = time.time()
        self.logger.info('Epoch: {0:d}, SELCON dataloader subset selection finished, takes {1:.4f}. '.format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights
