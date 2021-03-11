import apricot
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data.sampler import SubsetRandomSampler
import math


class CRAIGStrategy(DataSelectionStrategy):
    """
    Implementation of CRAIG Strategy from the paper :footcite:`mirzasoleiman2020coresets` for supervised learning frameworks.

    CRAIG strategy tries to solve the optimization problem given below for convex loss functions:

    .. math::
        \\sum_{i\\in \\mathcal{U}} \\min_{j \\in S, |S| \\leq k} \\| x^i - x^j \\|

    In the above equation, :math:`\\mathcal{U}` denotes the training set where :math:`(x^i, y^i)` denotes the :math:`i^{th}` training data point and label respectively,
    :math:`L_T` denotes the training loss, :math:`S` denotes the data subset selected at each round, and :math:`k` is the budget for the subset.

    Since, the above optimization problem is not dependent on model parameters, we run the subset selection only once right before the start of the training.

    CRAIG strategy tries to solve the optimization problem given below for non-convex loss functions:

    .. math::
        \\sum_{i\\in \\mathcal{U}} \\min_{j \\in S, |S| \\leq k} \\| \\nabla_{\\theta} {L_T}^i(\\theta) - \\nabla_{\\theta} {L_T}^j(\\theta) \\|

    In the above equation, :math:`\\mathcal{U}` denotes the training set, :math:`L_T` denotes the training loss, :math:`S` denotes the data subset selected at each round,
    and :math:`k` is the budget for the subset. In this case, CRAIG acts an adaptive subset selection strategy that selects a new subset every epoch.

    Both the optimization problems given above are an instance of facility location problems which is a submodular function. Hence, it can be optimally solved using greedy selection methods.

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
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    linear_layer: bool
        Apply linear transformation to the data
    if_convex: bool
        If convex or not
    selection_type: str
        Type of selection:
         - 'PerClass': PerClass Implementation where the facility location problem is solved for each class seperately for speed ups.
         - 'Supervised':  Supervised Implementation where the facility location problem is solved using a sparse similarity matrix by assigning the similarity of a point with other points of different class to zero.
    """

    def __init__(self, trainloader, valloader, model, loss,
                 device, num_classes, linear_layer, if_convex, selection_type, optimizer):
        """
        Constructer method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device)
        self.if_convex = if_convex
        self.selection_type = selection_type
        self.optimizer = optimizer

    def distance(self, x, y, exp=2):
        """
        Compute the distance.

        Parameters
        ----------
        x: Tensor
            First input tensor
        y: Tensor
            Second input tensor
        exp: float, optional
            The exponent value (default: 2)

        Returns
        ----------
        dist: Tensor
            Output tensor
        """

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.pow(x - y, exp).sum(2)
        # dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist

    def compute_score(self, model_params, idxs):
        """
        Compute the score of the indices.

        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        idxs: list
            The indices
        """

        trainset = self.trainloader.sampler.data_source
        subset_loader = torch.utils.data.DataLoader(trainset, batch_size=self.trainloader.batch_size, shuffle=False,
                                                    sampler=SubsetRandomSampler(idxs),
                                                    pin_memory=True)
        self.model.load_state_dict(model_params)
        self.N = 0
        g_is = []

        with torch.no_grad():
            if self.if_convex:
                for batch_idx, (inputs, targets) in enumerate(subset_loader):
                    inputs, targets = inputs, targets
                    if self.selection_type == 'PerBatch':
                        self.N += 1
                        g_is.append(inputs.view(inputs.size()[0], -1).mean(dim=0).view(1, -1))
                    else:
                        self.N += inputs.size()[0]
                        g_is.append(inputs.view(inputs.size()[0], -1))
            else:
                embDim = self.model.get_embedding_dim()
                for batch_idx, (inputs, targets) in enumerate(subset_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                    if self.selection_type == 'PerBatch':
                        self.N += 1
                    else:
                        self.N += inputs.size()[0]
                    out, l1 = self.model(inputs, freeze=True, last=True)
                    loss = self.loss(out, targets).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                        if self.selection_type == 'PerBatch':
                            g_is.append(torch.cat((l0_grads, l1_grads), dim=1).mean(dim=0).view(1, -1))
                        else:
                            g_is.append(torch.cat((l0_grads, l1_grads), dim=1))
                    else:
                        if self.selection_type == 'PerBatch':
                            g_is.append(l0_grads.mean(dim=0).view(1, -1))
                        else:
                            g_is.append(l0_grads)

            self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
            first_i = True
            if self.selection_type == 'PerBatch':
                g_is = torch.cat(g_is, dim=0)
                self.dist_mat = self.distance(g_is, g_is).cpu()
            else:
                for i, g_i in enumerate(g_is, 0):
                    if first_i:
                        size_b = g_i.size(0)
                        first_i = False
                    for j, g_j in enumerate(g_is, 0):
                        self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                        j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j).cpu()
        self.const = torch.max(self.dist_mat).item()
        self.dist_mat = (self.const - self.dist_mat).numpy()

    def compute_gamma(self, idxs):
        """
        Compute the gamma values for the indices.

        Parameters
        ----------
        idxs: list
            The indices

        Returns
        ----------
        gamma: list
            Gradient values of the input indices
        """

        if self.selection_type in ['PerClass', 'PerBatch']:
            gamma = [0 for i in range(len(idxs))]
            best = self.dist_mat[idxs]  # .to(self.device)
            rep = np.argmax(best, axis=0)
            for i in rep:
                gamma[i] += 1
        elif self.selection_type == 'Supervised':
            gamma = [0 for i in range(len(idxs))]
            best = self.dist_mat[idxs]  # .to(self.device)
            rep = np.argmax(best, axis=0)
            for i in range(rep.shape[1]):
                gamma[rep[0, i]] += 1
        return gamma

    def get_similarity_kernel(self):
        """
        Obtain the similarity kernel.

        Returns
        ----------
        kernel: ndarray
            Array of kernel values
        """
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                labels = targets
            else:
                tmp_target_i = targets
                labels = torch.cat((labels, tmp_target_i), dim=0)
        kernel = np.zeros((labels.shape[0], labels.shape[0]))
        for target in np.unique(labels):
            x = np.where(labels == target)[0]
            # prod = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
            for i in x:
                kernel[i, x] = 1
        return kernel

    def select(self, budget, model_params):
        """
        Data selection method using different submodular optimization
        functions.

        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters
        optimizer: str
            The optimization approach for data selection. Must be one of
            'random', 'modular', 'naive', 'lazy', 'approximate-lazy', 'two-stage',
            'stochastic', 'sample', 'greedi', 'bidirectional'

        Returns
        ----------
        total_greedy_list: list
            List containing indices of the best datapoints
        gammas: list
            List containing gradients of datapoints present in greedySet
        """

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                labels = targets
            else:
                tmp_target_i = targets
                labels = torch.cat((labels, tmp_target_i), dim=0)
        # per_class_bud = int(budget / self.num_classes)
        total_greedy_list = []
        gammas = []
        if self.selection_type == 'PerClass':
            for i in range(self.num_classes):
                idxs = torch.where(labels == i)[0]
                self.compute_score(model_params, idxs)
                fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=math.ceil(
                                                                                      budget * len(idxs) / self.N_trn),
                                                                                  optimizer=self.optimizer)
                sim_sub = fl.fit_transform(self.dist_mat)
                greedyList = list(np.argmax(sim_sub, axis=1))
                gamma = self.compute_gamma(greedyList)
                total_greedy_list.extend(idxs[greedyList])
                gammas.extend(gamma)
            rand_indices = np.random.permutation(len(total_greedy_list))
            total_greedy_list = list(np.array(total_greedy_list)[rand_indices])
            gammas = list(np.array(gammas)[rand_indices])
        elif self.selection_type == 'Supervised':
            for i in range(self.num_classes):
                if i == 0:
                    idxs = torch.where(labels == i)[0]
                    N = len(idxs)
                    self.compute_score(model_params, idxs)
                    row = idxs.repeat_interleave(N)
                    col = idxs.repeat(N)
                    data = self.dist_mat.flatten()
                else:
                    idxs = torch.where(labels == i)[0]
                    N = len(idxs)
                    self.compute_score(model_params, idxs)
                    row = torch.cat((row, idxs.repeat_interleave(N)), dim=0)
                    col = torch.cat((col, idxs.repeat(N)), dim=0)
                    data = np.concatenate([data, self.dist_mat.flatten()], axis=0)
            sparse_simmat = csr_matrix((data, (row.numpy(), col.numpy())), shape=(self.N_trn, self.N_trn))
            self.dist_mat = sparse_simmat
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=budget, optimizer=self.optimizer)
            sim_sub = fl.fit_transform(sparse_simmat)
            total_greedy_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
            gammas = self.compute_gamma(total_greedy_list)
        elif self.selection_type == 'PerBatch':
            idxs = torch.arange(self.N_trn)
            N = len(idxs)
            self.compute_score(model_params, idxs)
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=math.ceil(
                                                                                  budget / self.trainloader.batch_size),
                                                                              optimizer=self.optimizer)
            sim_sub = fl.fit_transform(self.dist_mat)
            temp_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
            gammas_temp = self.compute_gamma(temp_list)
            batch_wise_indices = list(self.trainloader.batch_sampler)
            for i in range(len(temp_list)):
                tmp = batch_wise_indices[temp_list[i]]
                total_greedy_list.extend(tmp)
                gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))
        return total_greedy_list, gammas
