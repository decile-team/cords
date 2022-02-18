from ..dssdataloader import DSSDataLoader


class NonAdaptiveDSSDataLoader(DSSDataLoader):
    """
    Implementation of NonAdaptiveDSSDataLoader class which serves as base class for dataloaders of other
    nonadaptive subset selection strategies for supervised learning setting.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary
    logger: class
        Logger for logging the information
    """
    def __init__(self, train_loader, val_loader, dss_args, logger, *args,
                 **kwargs):
        assert "device" in dss_args.keys(), "'device' is a compulsory argument. Include it as a key in dss_args"
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.initialized = False
        self.device = dss_args.device
        super(NonAdaptiveDSSDataLoader, self).__init__(train_loader.dataset, dss_args,
                                                       logger, *args, **kwargs)

    def __iter__(self):
        """
        Iter function that returns the iterator of the data subset loader.
        """
        return self.subset_loader.__iter__()



