from ..dssdataloader import DSSDataLoader


class NonAdaptiveDSSDataLoader(DSSDataLoader):
    def __init__(self, train_loader, val_loader, dss_args, logger, *args,
                 **kwargs):
        super(NonAdaptiveDSSDataLoader, self).__init__(train_loader.dataset, dss_args,
                                                       logger, *args, **kwargs)

        assert "device" in dss_args.keys(), "'device' is a compulsory argument. Include it as a key in dss_args"
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.initialized = False

    def __iter__(self):
        return self.subset_loader.__iter__()



