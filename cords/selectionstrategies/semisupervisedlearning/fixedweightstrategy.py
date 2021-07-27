import math
import time
import torch
import numpy as np
from .dataselectionstrategy import DataSelectionStrategy
from ..helpers import OrthogonalMP_REG_Parallel, OrthogonalMP_REG, OptimalWeights
from torch.utils.data import Subset, DataLoader


class FixedWeightStrategy(DataSelectionStrategy):
    """
    Parameters
	----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    loss_type: class
        The type of loss criterion
    eta: float
        Learning rate. Step size for the one step gradient update
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    linear_layer: bool
        Apply linear transformation to the data
    selection_type: str
        Type of selection -
        - 'PerClass': PerClass method is where OMP algorithm is applied on each class data points seperately.
        - 'PerBatch': PerBatch method is where OMP algorithm is applied on each minibatch data points.
        - 'PerClassPerGradient': PerClassPerGradient method is same as PerClass but we use the gradient corresponding to classification layer of that class only.
    valid : bool, optional
        If valid==True we use validation dataset gradient sum in OMP otherwise we use training dataset (default: False)
    lam : float
        Regularization constant of OMP solver
    eps : float
        Epsilon parameter to which the above optimization problem is solved using OMP algorithm
    """
    def __init__(self, trainloader, valloader, model, tea_model, ssl_alg, loss,
                 eta, device, num_classes, linear_layer, selection_type, valid=True, lam=0, eps=1e-4):
        """
        Constructor method
        """
        super().__init__(trainloader, valloader, model, tea_model, ssl_alg, num_classes, linear_layer, loss, device)
        self.eta = eta  # step size for the one step gradient update
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type
        self.valid = valid
        self.lam = lam
        self.eps = eps

    def optimalWeightsWrapper(self, X, Y, bud):

        if self.device == "cpu":
            ind, weights = OptimalWeights(X.cpu().numpy(), Y.cpu().numpy(), nnz=bud)

        else:
            ind, weights = OptimalWeights(X, Y, nnz=bud, device=self.device)

        return ind, weights

    def select(self, budget, model_params, tea_model_params):
        """
        Apply OMP Algorithm for data selection

        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing model's parameters
        tea_model_params: OrderedDict
            Python dictionary object containing teacher model's parameters
        Returns
        ----------
        idxs: list
            List containing indices of the best datapoints,
        gammas: weights tensors
            Tensor containing weights of each instance
        """
        omp_start_time = time.time()
        self.update_model(model_params, tea_model_params)

        if self.selection_type == 'PerClass':
            self.get_labels(valid=self.valid)
            idxs = []
            gammas = []
            for i in range(self.num_classes):
                trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()
                trn_data_sub = Subset(self.trainloader.dataset, trn_subset_idx)
                self.pctrainloader = DataLoader(trn_data_sub, batch_size=self.trainloader.batch_size,
                                                shuffle=False, pin_memory=True)
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
                    val_data_sub = Subset(self.valloader.dataset, val_subset_idx)
                    self.pcvalloader = DataLoader(val_data_sub, batch_size=self.trainloader.batch_size,
                                                  shuffle=False, pin_memory=True)

                self.compute_gradients(self.valid, batch=False, perClass=True)
                trn_gradients = self.grads_per_elem
                if self.valid:
                    sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
                else:
                    sum_val_grad = torch.sum(trn_gradients, dim=0)
                idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                                         sum_val_grad,
                                                         math.ceil(budget * len(trn_subset_idx) / self.N_trn))
                idxs.extend(list(np.array(trn_subset_idx)[idxs_temp]))
                gammas.extend(gammas_temp)

        elif self.selection_type == 'PerBatch':
            self.compute_gradients(self.valid, batch=True, perClass=False)
            idxs = []
            gammas = []
            trn_gradients = self.grads_per_elem
            if self.valid:
                sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
            else:
                sum_val_grad = torch.sum(trn_gradients, dim=0)
            idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                                     sum_val_grad, math.ceil(budget / self.trainloader.batch_size))
            batch_wise_indices = list(self.trainloader.batch_sampler)
            for i in range(len(idxs_temp)):
                tmp = batch_wise_indices[idxs_temp[i]]
                idxs.extend(tmp)
                gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))

        elif self.selection_type == 'PerClassPerGradient':
            self.get_labels(valid=self.valid)
            idxs = []
            gammas = []
            embDim = self.model.get_embedding_dim()
            for i in range(self.num_classes):
                trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()
                trn_data_sub = Subset(self.trainloader.dataset, trn_subset_idx)
                self.pctrainloader = DataLoader(trn_data_sub, batch_size=self.trainloader.batch_size,
                                                shuffle=False, pin_memory=True)
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
                    val_data_sub = Subset(self.valloader.dataset, val_subset_idx)
                    self.pcvalloader = DataLoader(val_data_sub, batch_size=self.trainloader.batch_size,
                                                  shuffle=False, pin_memory=True)
                self.compute_gradients(self.valid, batch=False, perClass=True)
                trn_gradients = self.grads_per_elem
                tmp_gradients = trn_gradients[:, i].view(-1, 1)
                tmp1_gradients = trn_gradients[:,
                                 self.num_classes + (embDim * i): self.num_classes + (embDim * (i + 1))]
                trn_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1)

                if self.valid:
                    val_gradients = self.val_grads_per_elem
                    tmp_gradients = val_gradients[:, i].view(-1, 1)
                    tmp1_gradients = val_gradients[:,
                                     self.num_classes + (embDim * i): self.num_classes + (embDim * (i + 1))]
                    val_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1)
                    sum_val_grad = torch.sum(val_gradients, dim=0)
                else:
                    sum_val_grad = torch.sum(trn_gradients, dim=0)

                idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                                         sum_val_grad,
                                                         math.ceil(budget * len(trn_subset_idx) / self.N_trn))
                idxs.extend(list(np.array(trn_subset_idx)[idxs_temp]))
                gammas.extend(gammas_temp)

        omp_end_time = time.time()
        diff = budget - len(idxs)

        if diff > 0:
            remainList = set(np.arange(self.N_trn)).difference(set(idxs))
            new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
            idxs.extend(new_idxs)
            gammas.extend([1 for _ in range(diff)])
            idxs = np.array(idxs)
            gammas = np.array(gammas)

        if self.selection_type in ["PerClass", "PerClassPerGradient"]:
            rand_indices = np.random.permutation(len(idxs))
            idxs = list(np.array(idxs)[rand_indices])
            gammas = list(np.array(gammas)[rand_indices])

        print("OMP algorithm Subset Selection time is: ", omp_end_time - omp_start_time)
        return idxs, gammas