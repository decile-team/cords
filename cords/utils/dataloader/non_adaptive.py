import copy
from abc import ABC

import numpy as np
import apricot
import math
import time

from cords.selectionstrategies.supervisedlearning import CRAIGStrategy
from cords.utils.dataloader.base import DSSDataLoader


class NonAdaptiveDSSDataLoader(DSSDataLoader):
    def __init__(self, train_loader, val_loader, budget, model, loss, device, verbose=False, *args,
                 **kwargs):
        super(NonAdaptiveDSSDataLoader, self).__init__(train_loader.dataset, budget,
                                                       verbose=verbose, *args, **kwargs)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss = copy.deepcopy(loss)
        self.device = device
        self.initialized = False

    def __iter__(self):
        return self.subset_loader.__iter__()


class RandomDataLoader(NonAdaptiveDSSDataLoader):
    pass


class SubmodDataLoader(NonAdaptiveDSSDataLoader):
    # Currently split dataset with size of |max_chunk| then proportionably select samples in every chunk
    # Otherwise distance matrix will be too large
    def __init__(self, train_loader, val_loader, budget, model, loss, device, size_chunk=2000, verbose=False, *args,
                 **kwargs):
        if size_chunk:
            print("You are using max_chunk: %s" % size_chunk)
        self.size_chunk = size_chunk
        super(SubmodDataLoader, self).__init__(train_loader, val_loader, budget, model, loss, device, verbose=verbose,
                                               *args,
                                               **kwargs)

    def _init_subset_indices(self):
        X = np.array([x for (x, _y) in self.dataset])
        m = X.shape[0]
        # Chunking dataset to calculate pairwise distance with limited memory
        sample_indices = []
        size_chunk, budget = self.size_chunk, self.budget
        n_chunks = math.ceil(m / self.size_chunk)
        budget_chunk = math.ceil(budget / n_chunks)
        for i_chunk in range(n_chunks):
            l_idx = i_chunk * size_chunk
            r_idx = min(m, (i_chunk + 1) * size_chunk)
            n_samples = min(budget_chunk, budget - len(sample_indices))
            chunk = X[l_idx: r_idx, :]
            _sample_indices = self._chunk_select(chunk, n_samples)
            _sample_indices = [_sample_indice + l_idx for _sample_indice in _sample_indices]
            sample_indices += _sample_indices
        return np.array(sample_indices)


# Submodular optimization based

class FacLocDataLoader(SubmodDataLoader):

    def _chunk_select(self, chunk, n_samples):
        f = apricot.functions.facilityLocation.FacilityLocationSelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)


class GraphCutDataLoader(SubmodDataLoader):

    def _chunk_select(self, chunk, n_samples):
        f = apricot.functions.graphCut.GraphCutSelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)


class SumRedundancyDataLoader(SubmodDataLoader):

    def _chunk_select(self, chunk, n_samples):
        f = apricot.functions.sumRedundancy.SumRedundancySelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)


class SaturatedCoverageDataLoader(SubmodDataLoader):

    def _chunk_select(self, chunk, n_samples):
        f = apricot.functions.facilityLocation.FacilityLocationSelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)


# class GLISTERDataLoader(AdaptiveDSSDataLoader):
#
#     def __init__(self, train_loader, val_loader, budget, select_every, model, loss, eta, device, num_cls,
#                  linear_layer, selection_type, r, verbose=True, *args, **kwargs):
#         super(GLISTERDataLoader, self).__init__(train_loader, val_loader, budget, select_every, model, loss,
#                                                 device,
#                                                 verbose=verbose, *args, **kwargs)
#         self.strategy = GLISTERStrategy(train_loader, val_loader, copy.deepcopy(model), loss, eta, device,
#                                         num_cls, linear_layer, selection_type, r=r)
#         self.train_model = model
#         self.eta = eta
#         self.num_cls = num_cls
#         if self.verbose:
#             print('Glister dataloader loader initialized. ')
#
#     def _resample_subset_indices(self):
#         if self.verbose:
#             start = time.time()
#             print('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
#         cached_state_dict = copy.deepcopy(self.train_model.state_dict())
#         clone_dict = copy.deepcopy(self.train_model.state_dict())
#         subset_indices, _ = self.strategy.select(self.budget, clone_dict)
#         self.train_model.load_state_dict(cached_state_dict)
#         if self.verbose:
#             end = time.time()
#             print('Epoch: {0:d}, subset selection finished, takes {1:.2f}. '.format(self.cur_epoch, (end - start)))
#         return subset_indices


class CRAIGDataLoader(NonAdaptiveDSSDataLoader):

    def __init__(self, train_loader, val_loader, budget, model, loss, device, num_cls,
                 linear_layer, if_convex, selection_type, optimizer, *args, **kwargs):
        super(CRAIGDataLoader, self).__init__(train_loader, val_loader, budget, model, loss, device, *args, **kwargs)
        self.strategy = CRAIGStrategy(train_loader, train_loader, model, loss,
                                      device, num_cls, linear_layer, if_convex,
                                      selection_type, optimizer=optimizer)
        self.train_model = model
        self.num_cls = num_cls
        if self.verbose:
            print('CRAIG dataloader loader initialized. ')

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
