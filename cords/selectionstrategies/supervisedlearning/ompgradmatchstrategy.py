import math
import time
import torch
import numpy as np
from .dataselectionstrategy import DataSelectionStrategy
from ..helpers import OrthogonalMP_REG_Parallel, OrthogonalMP_REG

class OMPGradMatchStrategy(DataSelectionStrategy):

    """
    Implementation of OMPGradMatch Strategy from the paper :footcite:`sivasubramanian2020gradmatch` for supervised learning frameworks.

    OMPGradMatch strategy tries to solve the optimization problem given below:

    .. math::
        \\min_{\\mathbf{w}, S: |S| \\leq k} \\Vert \\sum_{i \\in S} w_i \\nabla_{\\theta}L_T^i(\\theta) -  \\nabla_{\\theta}L(\\theta)\\Vert

    In the above equation, :math:`\\mathbf{w}` denotes the weight vector that contains the weights for each data instance, :math:`\mathcal{U}` training set where :math:`(x^i, y^i)` denotes the :math:`i^{th}` training data point and label respectively,
    :math:`L_T` denotes the training loss, :math:`L` denotes either training loss or validation loss depending on the parameter valid,
    :math:`S` denotes the data subset selected at each round, and :math:`k` is the budget for the subset.

    The above optimization problem is solved using the Orthogonal Matching Pursuit(OMP) algorithm.

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
        - 'PerClassPerGradient': PerClassPerGradient method is same as PerClass but we use the gradient corresponding to classification layer of that class only.
    valid : bool, optional
        If valid==True we use validation dataset gradient sum in OMP otherwise we use training dataset (default: False)
    lam : float
        Regularization constant of OMP solver
    eps : float
        Epsilon parameter to which the above optimization problem is solved using OMP algorithm
    """

    def __init__(self, trainloader, valloader, model, loss_type,
                 eta, device, num_classes, linear_layer, selection_type, valid=True, lam=0, eps=1e-4, r=1):
        """
        Constructor method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer)
        self.loss_type = loss_type
        self.eta = eta  # step size for the one step gradient update
        self.device = device
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type
        self.valid = valid
        self.lam = lam
        self.eps = eps

    def ompwrapper(self, X, Y, bud):
        if self.device == "cpu":
            reg = OrthogonalMP_REG(X.cpu().numpy(), Y.cpu().numpy(), nnz=bud, positive=True, lam=0)
            ind = np.nonzero(reg)[0]
        else:
            reg = OrthogonalMP_REG_Parallel(X, Y, nnz=bud,
                                          positive=True, lam=self.lam,
                                          tol=self.eps, device=self.device)
            ind = torch.nonzero(reg).view(-1)
        return ind.tolist(), reg[ind].tolist()

    def select(self, budget, model_params):
        """
        Apply OMP Algorithm for data selection

        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters

        Returns
        ----------
        greedySet: list
            List containing indices of the best datapoints,
        budget: Tensor
            Tensor containing gradients of datapoints present in greedySet
        """
        omp_start_time = time.time()
        self.update_model(model_params)
        start_time = time.time()
        self.compute_gradients(self.valid)
        end_time = time.time()
        print("Per Element gradient computation time is: ", end_time - start_time)
        if self.selection_type == 'PerClass':
            idxs = []
            gammas = []
            for i in range(self.num_classes):
                trn_subset_idx = torch.where(self.trn_lbls == i)
                trn_gradients = self.grads_per_elem[trn_subset_idx]
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)
                    sum_val_grad = torch.sum(self.val_grads_per_elem[val_subset_idx], dim=0)
                else:
                    sum_val_grad = torch.sum(trn_gradients, dim=0)
                idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                          sum_val_grad, math.ceil(budget * len(trn_subset_idx[0]) / self.N_trn))
                idxs.extend(list(trn_subset_idx[0].cpu().numpy()[idxs_temp]))
                gammas.extend(gammas_temp)
        elif self.selection_type == 'PerClassPerGradient':
            idxs = []
            gammas = []
            embDim = self.model.get_embedding_dim()
            for i in range(self.num_classes):
                trn_subset_idx = torch.where(self.trn_lbls == i)
                trn_gradients = self.grads_per_elem[trn_subset_idx]
                tmp_gradients = trn_gradients[:, i].view(-1, 1)
                tmp1_gradients = trn_gradients[: , self.num_classes + (embDim * i) : self.num_classes + (embDim * (i + 1))]
                trn_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1)
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)
                    val_gradients = self.val_grads_per_elem[val_subset_idx]
                    tmp_gradients = val_gradients[:, i].view(-1, 1)
                    tmp1_gradients = val_gradients[:,
                                     self.num_classes + (embDim * i): self.num_classes + (embDim * (i + 1))]
                    val_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1)
                    sum_val_grad = torch.sum(val_gradients, dim=0)
                else:
                    sum_val_grad = torch.sum(trn_gradients, dim=0)
                idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                                         sum_val_grad,
                                                         math.ceil(budget * len(trn_subset_idx[0]) / self.N_trn))
                idxs.extend(list(trn_subset_idx[0].cpu().numpy()[idxs_temp]))
                gammas.extend(gammas_temp)
        omp_end_time = time.time()
        diff = budget - len(idxs)
        remainList = set(np.arange(self.N_trn)).difference(set(idxs))
        new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
        idxs.extend(new_idxs)
        gammas.extend([1 for _ in range(diff)])
        idxs = np.array(idxs)
        gammas = np.array(gammas)
        rand_indices = np.random.permutation(len(idxs))
        idxs = list(idxs[rand_indices])
        gammas = list(gammas[rand_indices])
        print("OMP algorithm Subset Selection time is: ", omp_end_time - omp_start_time)
        return idxs, gammas