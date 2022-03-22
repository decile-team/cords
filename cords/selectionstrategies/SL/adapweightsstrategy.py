import math
import time
import torch
import numpy as np
from .dataselectionstrategy import DataSelectionStrategy
# from ..helpers import OrthogonalMP_REG_Parallel, OrthogonalMP_REG, OrthogonalMP_REG_Parallel_V1
from torch.utils.data import Subset, DataLoader
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression

class AdapWeightsStrategy(DataSelectionStrategy):
    """
    Implementation of GradMatch Strategy from the paper :footcite:`sivasubramanian2020gradmatch` for supervised learning frameworks.

    GradMatch strategy tries to solve the optimization problem given below:

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
    loss: class
        PyTorch loss function for training
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
    logger : class
        - logger object for logging the information
    valid : bool
        If valid==True, we use validation dataset gradient sum in OMP otherwise we use training dataset (default: False)
    """

    def __init__(self, trainloader, valloader, model, loss,
                 eta, device, num_classes, linear_layer,
                 selection_type, logger, ss_indices, valid=False):
        """
        Constructor method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device, logger)
        self.eta = eta  # step size for the one step gradient update
        self.device = device
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type
        self.valid = valid

        self.ss_indices = ss_indices

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
        idxs: list
            List containing indices of the best datapoints,
        gammas: weights tensors
            Tensor containing weights of each instance
        """
        omp_start_time = time.time()
        self.update_model(model_params)

        self.compute_gradients(self.valid, perBatch=False, perClass=False)

        idxs = self.ss_indices
        trn_gradients = self.grads_per_elem

        ss_grad = trn_gradients[idxs].clone().detach()
        ss_grad = torch.transpose(ss_grad, 0, 1)
        b_ = trn_gradients.sum(dim = 0)
        if self.valid:
                b_ = torch.sum(self.val_grads_per_elem, dim=0)
        print("ss_grad shape in adapweightsstrategy:", ss_grad.shape)
        
        if ss_grad.shape[0] != b_.shape[0]:
            print("Shapes mismatch, error in adapweightstrategy, exiting")
            exit(1)
        elif ss_grad.shape[1] > 0 and b_.shape[0] > 0:
            # reg_nnls = LinearRegression(positive=True)
            # gammas = reg_nnls.fit(np.nan_to_num(ss_grad.detach().cpu().numpy()), np.nan_to_num(b_.detach().cpu().numpy())).coef_
            gammas, _ = nnls(np.nan_to_num(ss_grad.detach().cpu().numpy()), np.nan_to_num(b_.detach().cpu().numpy())\
                , maxiter = int(30*ss_grad.shape[1]))
        else:
            gammas = list(np.random.ranint(1,10,ss_grad.shape[1]))

        omp_end_time = time.time()
        self.logger.debug("AdapWeights algorithm Subset Selection time is: %.4f", omp_end_time - omp_start_time)
        return idxs, torch.FloatTensor(gammas)
