from abc import abstractmethod
from torch.utils.data import Subset, Dataset
from torch.utils.data.dataloader import DataLoader
from cords.selectionstrategies.supervisedlearning import GLISTERStrategy, RandomStrategy
import copy, logging, time


class DSSDataset(Dataset):
    def __init__(self, full_data):
        self.full_data = full_data
        self.indices, self.data = None, None

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def reload(self, indices):
        self.indices = indices
        self.data = Subset(self.full_data, self.indices)


class DSSDataLoader:
    def __init__(self, full_data, verbose=False, *args, **kwargs):
        super(DSSDataLoader, self).__init__()
        # TODO: Integrate verbose in logging
        self.verbose = verbose
        self.dataset = DSSDataset(full_data)
        self.subset_loader = DataLoader(self.dataset, *args, **kwargs)

    def __getattr__(self, item):
        self.subset_loader.__getattribute__(item)

    @abstractmethod
    def _resample(self):
        raise Exception('Not implemented. ')

    @abstractmethod
    def state_dict(self):
        # data, subset, index
        pass

    @abstractmethod
    def load_state_dict(self):
        pass


class OnlineDSSDataLoader(DSSDataLoader):
    def __init__(self, train_loader, val_loader, budget, select_every, model, loss, device, verbose=False, *args,
                 **kwargs):
        super(OnlineDSSDataLoader, self).__init__(train_loader.dataset, verbose=verbose, *args, **kwargs)
        self.cur_epoch = 1
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.budget = budget
        self.select_every = select_every
        self.model = model
        self.loss = loss
        self.device = device
        self.subset_idxes = None
        self.strategy = None

    def __iter__(self):
        if self.verbose:
            logging.info('Epoch: {0:d}, reading data... '.format(self.cur_epoch))
        if self.cur_epoch % self.select_every == 0:
            self._resample()
        if self.verbose:
            logging.info('Epoch: {0:d}, finished reading data. '.format(self.cur_epoch))
        self.cur_epoch += 1
        # breakpoint()
        return self.subset_loader.__iter__()

    def _finalize_initialization(self):
        self._resample()

    @abstractmethod
    def _resample(self):
        raise Exception('Not implemented. ')

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass


# GLISTER
class GLISTERDataLoader(OnlineDSSDataLoader):

    def __init__(self, train_loader, val_loader, budget, select_every, model, loss, eta, device, num_cls,
                 linear_layer, selection_type, r, batch_size, verbose=True, *args, **kwargs):
        super(GLISTERDataLoader, self).__init__(train_loader, val_loader, budget, select_every, model, loss, device,
                                                verbose=verbose, *args, **kwargs)
        self.strategy = GLISTERStrategy(train_loader, val_loader, copy.deepcopy(model), loss, eta, device,
                                        num_cls, linear_layer, selection_type, r=r, verbose=verbose)
        self.eta = eta
        self.num_cls = num_cls
        self.batch_size = batch_size
        super(GLISTERDataLoader, self)._finalize_initialization()
        if self.verbose:
            logging.info('Glister data loader initialized. ')

    def _resample(self):
        if self.verbose:
            start = time.time()
            logging.info('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        cached_state_dict = copy.deepcopy(self.model.state_dict())
        clone_dict = copy.deepcopy(self.model.state_dict())
        subset_idxes, _ = self.strategy.select(self.budget, clone_dict)
        self.model.load_state_dict(cached_state_dict)
        self.subset_idxes = subset_idxes
        self.dataset.reload(subset_idxes)
        if self.verbose:
            end = time.time()
            logging.info(
                'Epoch: {0:d}, subset selection finished, takes {1:.2f}. '.format(self.cur_epoch, (end - start)))


# Random
class OnlineRandomDataLoader(OnlineDSSDataLoader):
    def __init__(self, train_loader, val_loader, budget, select_every, model, loss, device, verbose=True, *args,
                 **kwargs):
        super(OnlineRandomDataLoader, self).__init__(train_loader, val_loader, budget, select_every, model, loss,
                                                     device, verbose=verbose, *args,
                                                     **kwargs)
        self.strategy = RandomStrategy(train_loader, online=True)
        super(OnlineRandomDataLoader, self)._finalize_initialization()

    def _resample(self):
        subset_idxes, _ = self.strategy.select(self.budget)
        self.subset_idxes = subset_idxes
        self.dataset.reload(subset_idxes)
