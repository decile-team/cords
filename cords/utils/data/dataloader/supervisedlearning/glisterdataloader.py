from abc import abstractmethod
from .adaptivedssdataloader import AdaptiveDSSDataLoader
import logging
from cords.selectionstrategies.supervisedlearning import GLISTERStrategy
import time, copy, logging


# GLISTER
class GLISTERDataLoader(AdaptiveDSSDataLoader):

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


# # Random
# class OnlineRandomDataLoader(OnlineDSSDataLoader):
#     def __init__(self, train_loader, val_loader, select_ratio, select_every, model, loss, device, verbose=True, *args,
#                  **kwargs):
#         super(OnlineRandomDataLoader, self).__init__(train_loader, val_loader, select_ratio, select_every, model, loss,
#                                                      device, verbose=verbose, *args,
#                                                      **kwargs)
#         self.strategy = RandomStrategy(train_loader, online=True)

#     def _resample_subset_indices(self):
#         if self.verbose:
#             start = time.time()
#             logging.info('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
#         logging.debug("Random budget: %d", self.budget)
#         subset_indices, _ = self.strategy.select(self.budget)
#         if self.verbose:
#             end = time.time()
#             logging.info(
#                 'Epoch: {0:d}, subset selection finished, takes {1:.2f}. '.format(self.cur_epoch, (end - start)))
#         return subset_indices