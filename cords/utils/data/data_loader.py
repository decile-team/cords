from abc import abstractmethod
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from cords.selectionstrategies.supervisedlearning import GLISTERStrategy, RandomStrategy
import copy, logging, time
import numpy as np


class DSSDataLoader:
    def __init__(self, full_data, budget, verbose=False, *args, **kwargs):
        super(DSSDataLoader, self).__init__()
        # TODO: Integrate verbose in logging
        self.verbose = verbose
        # self.full_data = full_data
        self.dataset = full_data
        self.budget = budget
        self.subset_loader = None
        # Subset
        self.strategy = None
        self.loader_args = args
        self.loader_kwargs = kwargs
        self._init()

    def __getattr__(self, item):
        self.subset_loader.__getattribute__(item)

    def resample(self):
        subset_indices = self._resample_subset_indices()
        logging.debug("Subset indices length: %d", len(subset_indices))
        self._refresh_subset_loader(subset_indices)
        logging.debug("Subset loader inited, args: %s, kwargs: %s", self.loader_args, self.loader_kwargs)
        logging.info('Sample finished, total number of data: %d, number of subset: %d', len(self.dataset), len(self.subset_loader.dataset))

    def _init(self):
        random_indices = np.random.choice(len(self.dataset), size=self.budget, replace=False)
        self._refresh_subset_loader(random_indices)

    def _refresh_subset_loader(self, indices):
        self.subset_loader = DataLoader(Subset(self.dataset, indices), *self.loader_args, **self.loader_kwargs)

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self):
        pass


class OnlineDSSDataLoader(DSSDataLoader):
    def __init__(self, train_loader, val_loader, select_ratio, select_every, model, loss, device, verbose=False, *args,
                 **kwargs):
        super(OnlineDSSDataLoader, self).__init__(train_loader.dataset, int(select_ratio*len(train_loader.dataset)),
                                                  verbose=verbose, *args, **kwargs)
        self.cur_epoch = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.select_every = select_every
        self.model = model
        self.loss = loss
        self.device = device

    def __iter__(self):
        if self.verbose:
            logging.info('Epoch: {0:d}, reading data... '.format(self.cur_epoch))
        if self.cur_epoch > 0 and self.cur_epoch % self.select_every == 0:
        # if self.cur_epoch % self.select_every == 0:
            self.resample()
        if self.verbose:
            logging.info('Epoch: {0:d}, finished reading data. '.format(self.cur_epoch))
        self.cur_epoch += 1
        return self.subset_loader.__iter__()

    @abstractmethod
    def _resample_subset_indices(self):
        raise Exception('Not implemented. ')

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass


# GLISTER
class GLISTERDataLoader(OnlineDSSDataLoader):

    def __init__(self, train_loader, val_loader, select_ratio, select_every, model, loss, eta, device, num_cls,
                 linear_layer, selection_type, r, verbose=True, *args, **kwargs):
        super(GLISTERDataLoader, self).__init__(train_loader, val_loader, select_ratio, select_every, model, loss, device,
                                                verbose=verbose, *args, **kwargs)
        self.strategy = GLISTERStrategy(train_loader, val_loader, copy.deepcopy(model), loss, eta, device,
                                        num_cls, linear_layer, selection_type, r=r, verbose=verbose)
        self.train_model = model
        self.eta = eta
        self.num_cls = num_cls
        if self.verbose:
            logging.info('Glister data loader initialized. ')

    def _resample_subset_indices(self):
        if self.verbose:
            start = time.time()
            logging.info('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, _ = self.strategy.select(self.budget, clone_dict)
        self.train_model.load_state_dict(cached_state_dict)
        if self.verbose:
            end = time.time()
            logging.info(
                'Epoch: {0:d}, subset selection finished, takes {1:.2f}. '.format(self.cur_epoch, (end - start)))
        return subset_indices


# # Grad-Match
# class GradMatchDataLoader(OnlineDSSDataLoader):
#
#     def __init__(self):
#         super(GLISTERDataLoader, self).__init__(train_loader, val_loader, select_ratio, select_every, model, loss, device,
#                                                 verbose=verbose, *args, **kwargs)
#         self.train_model = model
#
#     def _resample_subset_indices(self):
#         pass


# Random
class OnlineRandomDataLoader(OnlineDSSDataLoader):
    def __init__(self, train_loader, val_loader, select_ratio, select_every, model, loss, device, verbose=True, *args,
                 **kwargs):
        super(OnlineRandomDataLoader, self).__init__(train_loader, val_loader, select_ratio, select_every, model, loss,
                                                     device, verbose=verbose, *args,
                                                     **kwargs)
        self.strategy = RandomStrategy(train_loader, online=True)

    def _resample_subset_indices(self):
        if self.verbose:
            start = time.time()
            logging.info('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        logging.debug("Random budget: %d", self.budget)
        subset_indices, _ = self.strategy.select(self.budget)
        if self.verbose:
            end = time.time()
            logging.info(
                'Epoch: {0:d}, subset selection finished, takes {1:.2f}. '.format(self.cur_epoch, (end - start)))
        return subset_indices
