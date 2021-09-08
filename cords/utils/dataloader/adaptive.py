import copy
import logging
import time
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader

from cords.selectionstrategies.supervisedlearning import GLISTERStrategy, OMPGradMatchStrategy, RandomStrategy
from cords.utils.dataloader.base import DSSDataLoader
from cords.utils.utils import copy_


class AdaptiveDSSDataLoader(DSSDataLoader):
    def __init__(self, train_loader, val_loader, budget, select_every, model, loss, device, verbose=False, *args,
                 **kwargs):
        super(AdaptiveDSSDataLoader, self).__init__(train_loader.dataset, budget,
                                                    verbose=verbose, *args, **kwargs)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.select_every = select_every
        self.model = model
        self.loss = copy.deepcopy(loss)
        self.device = device
        self.initialized = False
        self.n_full_epoch, self.select_after = 0, 0

    def __iter__(self):
        self.initialized = True
        if self.n_full_epoch <= self.cur_epoch < self.select_after:
            logging.info(
                "Skipping epoch {0:d} due to warm-start option. ".format(self.cur_epoch, self.n_full_epoch))
            loader = DataLoader([])
        else:
            if self.verbose:
                logging.info('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
            if self.cur_epoch < self.n_full_epoch:
                loader = self.fullset_loader
            else:
                if self.cur_epoch % self.select_every == 0:
                    self.resample()
                loader = self.subset_loader
            if self.verbose:
                logging.info('Epoch: {0:d}, finished reading dataloader. '.format(self.cur_epoch))
        self.cur_epoch += 1
        return loader.__iter__()

    def set_warm(self, select_after, n_full_epoch):
        if self.initialized:
            raise Exception('DataLoader already Initialized. ')
        else:
            self.n_full_epoch, self.select_after = n_full_epoch, select_after

    @abstractmethod
    def _resample_subset_indices(self):
        raise Exception('Not implemented. ')

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass


class GLISTERDataLoader(AdaptiveDSSDataLoader):

    def __init__(self, train_loader, val_loader, budget, select_every, model, loss, eta, device, num_cls,
                 linear_layer, selection_type, r, verbose=True, *args, **kwargs):
        super(GLISTERDataLoader, self).__init__(train_loader, val_loader, budget, select_every, model, loss,
                                                device,
                                                verbose=verbose, *args, **kwargs)
        self.strategy = GLISTERStrategy(train_loader, val_loader, copy.deepcopy(model), loss, eta, device,
                                        num_cls, linear_layer, selection_type, r=r)
        self.train_model = model
        self.eta = eta
        self.num_cls = num_cls
        if self.verbose:
            print('Glister dataloader loader initialized. ')

    def _resample_subset_indices(self):
        if self.verbose:
            start = time.time()
            print('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, _ = self.strategy.select(self.budget, clone_dict)
        self.train_model.load_state_dict(cached_state_dict)
        if self.verbose:
            end = time.time()
            print('Epoch: {0:d}, subset selection finished, takes {1:.2f}. '.format(self.cur_epoch, (end - start)))
        return subset_indices


class OMPGradMatchDataLoader(AdaptiveDSSDataLoader):

    def __init__(self, train_loader, val_loader, budget, select_every, model, loss, eta, device, num_cls,
                 linear_layer, selection_type, valid, lam, eps, verbose=True, *args, **kwargs):
        assert loss.reduction == "none", "When using GradMatch, please set 'reduction' of loss function to 'none'. "
        loss = copy.deepcopy(loss)
        super(OMPGradMatchDataLoader, self).__init__(train_loader, val_loader, budget, select_every, model, loss,
                                                     device,
                                                     verbose=verbose, *args, **kwargs)
        self.strategy = OMPGradMatchStrategy(train_loader, val_loader, copy.deepcopy(model), loss, eta, device,
                                             num_cls, linear_layer, selection_type, valid, lam, eps, verbose=verbose)
        self.gamma = torch.ones(len(self.subset_loader.dataset)).to(device)
        self.train_model = model
        self.eta = eta
        self.num_cls = num_cls
        # TODO: Refactor weights non-invasively (maybe using hook).
        if self.verbose:
            print('Grad-match dataloader loader initialized. ')

    def _resample_subset_indices(self):
        if self.verbose:
            start = time.time()
            print("Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch))
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, gamma = self.strategy.select(self.budget, clone_dict)
        copy_(self.gamma, gamma)
        self.train_model.load_state_dict(cached_state_dict)
        if self.verbose:
            end = time.time()
            print("Epoch: {0:d}, subset selection finished, takes {1:.2f}. ".format(self.cur_epoch, (end - start)))
        return subset_indices


class OnlineRandomDataLoader(AdaptiveDSSDataLoader):
    def __init__(self, train_loader, val_loader, budget, select_every, model, loss, device, verbose=True, *args,
                 **kwargs):
        super(OnlineRandomDataLoader, self).__init__(train_loader, val_loader, budget, select_every, model, loss,
                                                     device, verbose=verbose, *args,
                                                     **kwargs)
        self.strategy = RandomStrategy(train_loader, online=True)

    def _resample_subset_indices(self):
        if self.verbose:
            start = time.time()
            print("Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch))
        logging.debug("Random budget: %d", self.budget)
        subset_indices, _ = self.strategy.select(self.budget)
        if self.verbose:
            end = time.time()
            print("Epoch: {0:d}, subset selection finished, takes {1:.2f}. ".format(self.cur_epoch, (end - start)))
        return subset_indices
