import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import apricot
from selectionstrategies.supervisedlearning.dataselectionstrategy import DataSelectionStrategy
from scipy.sparse import csr_matrix


class CRAIGStrategy(DataSelectionStrategy):
    """
    This class extends :class:`selectionstrategies.supervisedlearning.dataselectionstrategy.DataSelectionStrategy`
    to include submodular optmization functions using apricot for data selection.

    :param trainloader: Loading the training data using pytorch DataLoader
    :type trainloader: class
    :param valloader: Loading the validation data using pytorch DataLoader
    :type valloader: class
    :param model: Model architecture used for training
    :type model: class
    :param loss_type: The type of loss criterion
    :type loss_type: class
    :param device: The device being utilized - cpu | cuda
    :type device: str
    :param num_classes: The number of target classes in the dataset
    :type num_classes: int
    :param linear_layer: Apply linear transformation to the data
    :type linear_layer: bool
    :param if_convex: If convex or not
    :type if_convex: bool
    :param selection_type: PerClass or Supervised
    :type selection_type: str
    """

    def __init__(self, trainloader, valloader, model, loss_type,
                 device, num_classes, linear_layer, if_convex, selection_type):
        """
        Constructer method
        """

        super().__init__(trainloader, valloader, model, linear_layer)

        self.loss_type = loss_type  # Make sure it has reduction='none' instead of default
        self.device = device
        self.num_classes = num_classes
        self.if_convex = if_convex
        self.selection_type = selection_type


    def distance(self, x, y, exp=2):
        """
        Compute the distance.
 
        :param x: first input tensor
        :type x: Tensor
        :param y: second input tensor
        :type y: Tensor
        :param exp: The exponent value, defaults to 2
        :type exp: float, optional
        :return: Output tensor 
        :rtype: Tensor
        """

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.pow(x - y, exp).sum(2)
        #dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist


    def compute_score(self, model_params, idxs):
        """
        Compute the score of the indices.
        :param model_params: Python dictionary object containing models parameters
        :type model_params: OrderedDict
        :param idxs: The indices
        :type idxs: list
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
                    self.N += inputs.size()[0]
                    g_is.append(inputs.view(inputs.size()[0], -1))
            else:
                embDim = self.model.get_embedding_dim()
                for batch_idx, (inputs, targets) in enumerate(subset_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                    self.N += inputs.size()[0]
                    with torch.no_grad():
                        out, l1 = self.model(inputs, last=True)
                        data = F.softmax(out, dim=1)
                    outputs = torch.zeros(len(inputs), self.num_classes).to(self.device)
                    outputs.scatter_(1, targets.view(-1, 1), 1)
                    l0_grads = data - outputs
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                        g_is.append(torch.cat((l0_grads, l1_grads), dim=1))
                    else:
                        g_is.append(l0_grads)

            self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j)
        self.const = torch.max(self.dist_mat).item()
        self.dist_mat = (self.const - self.dist_mat).numpy()


    def compute_gamma(self, idxs):
        """
        Compute the gamma values for the indices.
        :param idxs: The indices
        :type idxs: list
        :return: gamma values 
        :rtype: list
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
        :return: array of kernel values
        :rtype: ndarray
        """

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                tmp_target = targets
            else:
                tmp_target_i = targets
                targets = torch.cat((tmp_target, tmp_target_i), dim=0)
        kernel = np.zeros((targets.shape[0], targets.shape[0]))
        for target in np.unique(targets):
            x = np.where(targets == target)[0]
            # prod = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
            for i in x:
                kernel[i, x] = 1
        return kernel


    def select(self, budget, model_params, optimizer):
        """
        Data selection method using different submodular optimization
        functions.
 
        :param budget: The number of data points to be selected
        :type budget: int
        :param model_params: Python dictionary object containing models parameters
        :type model_params: OrderedDict
        :param optimizer: The list of submodular functions to mix together
        :type optimizer: list  
        :return: List containing indices of the best datapoints, 
                list containing gradients of datapoints present in greedySet
        :rtype: list, list
        """

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                labels = targets
            else:
                tmp_target_i = targets
                labels = torch.cat((labels, tmp_target_i), dim=0)
        per_class_bud = int(budget / self.num_classes)
        total_greedy_list = []
        gammas = []
        if self.selection_type == 'PerClass':
            for i in range(self.num_classes):
                idxs = torch.where(labels == i)[0]
                self.compute_score(model_params, idxs)
                fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=per_class_bud,
                                                                                  optimizer=optimizer)
                sim_sub = fl.fit_transform(self.dist_mat)
                greedyList = list(np.argmax(sim_sub, axis=1))
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
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=budget,
                                                                              optimizer=optimizer)
            sim_sub = fl.fit_transform(sparse_simmat)
            total_greedy_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
            gammas = self.compute_gamma(total_greedy_list)
        return total_greedy_list, gammas
