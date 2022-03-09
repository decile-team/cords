import submodlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data.sampler import SubsetRandomSampler


class SubmodularSelectionStrategy(DataSelectionStrategy):
    """
    This class extends :class:`selectionstrategies.supervisedlearning.dataselectionstrategy.DataSelectionStrategy`
    to include submodular optmization functions using submodlib for data selection.

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
        PerClass or Supervised
    submod_func_type: str
        The type of submodular optimization function. Must be one of
        'facility-location', 'graph-cut', or 'saturated-coverage'
    optimizer: str
        The optimizer used to compute the optimal subset. Can be 'NaiveGreedy', 'StochasticGreedy', 'LazyGreedy', or 'LazierThanLazyGreedy'.
    num_neighbors: int, default=5
        Number of neighbors applicable for the sparse similarity kernel.
    """

    def __init__(self, trainloader, valloader, model, loss,
                 device, num_classes, linear_layer, if_convex, selection_type, submod_func_type, optimizer, num_neighbors=5):
        """
        Constructer method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device)
        self.if_convex = if_convex
        self.selection_type = selection_type
        self.submod_func_type = submod_func_type
        self.optimizer = optimizer
        self.num_neighbors = num_neighbors

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

        if self.selection_type == 'PerClass':
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

        Returns
        ----------
        total_greedy_list: list
            List containing indices of the best datapoints
        gammas: list
            List containing gradients of datapoints present in greedySet
        """

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                x_trn, labels = inputs, targets
            else:
                tmp_inputs, tmp_target_i = inputs, targets
                labels = torch.cat((labels, tmp_target_i), dim=0)
        per_class_bud = int(budget / self.num_classes)
        total_greedy_list = []
        gammas = []
        if self.selection_type == 'PerClass':
            for i in range(self.num_classes):
                idxs = torch.where(labels == i)[0]
                self.compute_score(model_params, idxs)
                fl_functions = {'facility-location':submodlib.functions.facilityLocation.FacilityLocationFunction,
                                'graph-cut':submodlib.functions.graphCut.graphCutFunction,
                                'saturated-coverage':submodlib.functions.saturatedCoverage.saturatedCoverageFunction}
                fl = fl_functions[self.submod_func_type](n=self.dist_mat.shape[0], mode='dense', sijs=self.dist_mat)
                greedyList = [idx for idx, _ in fl.maximize(budget=per_class_bud, optimizer=self.optimizer)]
                gamma = self.compute_gamma(greedyList)
                total_greedy_list.extend(idxs[greedyList])
                gammas.extend(gamma)

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
            fl_functions = {'facility-location':submodlib.functions.facilityLocation.FacilityLocationFunction,
                            'graph-cut':submodlib.functions.graphCut.graphCutFunction,
                            'saturated-coverage':submodlib.functions.saturatedCoverage.saturatedCoverageFunction}
            fl = fl_functions[self.submod_func_type](n=self.N_trn, mode='sparse', sijs=sparse_simmat, num_neighbors=self.num_neighbors)
            greedy_list = fl.maximize(budget=per_class_bud, optimizer=self.optimizer)
            gammas = self.compute_gamma(greedy_list)
        return total_greedy_list, gammas
