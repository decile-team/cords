from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import SELCONstrategy
import time, copy


# SELCONstrategy
class SELCONDataLoader(AdaptiveDSSDataLoader):
    def __init__(self, trainset, validset, train_loader, val_loader, dss_args, logger, *args, **kwargs):
        """
         Arguments assertion check
        """
        assert "model" in dss_args.keys(), "'model' is a compulsory argument. Include it as a key in dss_args"
        assert "loss" in dss_args.keys(), "'loss' is a compulsory argument. Include it as a key in dss_args"
        if dss_args.loss.reduction != "none":
            raise ValueError("Please set 'reduction' of loss function to 'none' for adaptive subset selection strategies")
        assert "device" in dss_args.keys(), "'device' is a compulsory argument. Include it as a key in dss_args"
        assert "num_classes" in dss_args.keys(), "'num_classes' is a compulsory argument for SELCON. Include it as a key in dss_args"
        assert "delta" in dss_args.keys(), "'delta' is a compulsory argument for SELCON. Include it as a key in dss_args"
        assert "linear_layer" in dss_args.keys(), "'linear_layer' is a compulsory argument for SELCON. Include it as a key in dss_args"
        
        '''
        self, trainloader, valloader, model, 
        loss_func, device, num_classes, delta, 
        linear_layer, lam, lr, logger, optimizer, 
        batch_size, criterion
        '''
        
        super(SELCONDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                                logger, *args, **kwargs)
        
        self.strategy = SELCONstrategy(trainset, validset, train_loader, val_loader, copy.deepcopy(dss_args.model), dss_args.loss, dss_args.device,
                                        dss_args.num_classes, dss_args.delta, dss_args.num_epochs, dss_args.linear_layer, dss_args.lam, dss_args.lr, 
                                        dss_args.logger, dss_args.optimizer, dss_args.batch_size, dss_args.criterion)
        self.train_model = dss_args.model
        self.logger.debug('SELCON dataloader initialized. ')

    def _resample_subset_indices(self):
        start = time.time()
        self.logger.debug('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict)
        self.train_model.load_state_dict(cached_state_dict)
        end = time.time()
        self.logger.info('Epoch: {0:d}, SELCON dataloader subset selection finished, takes {1:.4f}. '.format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights
