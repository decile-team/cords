import math
import random
import time
import torch
import torch.nn.functional as F
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data import Subset, DataLoader
import numpy as np


class RETRIEVEStrategy(DataSelectionStrategy):
    """
    Implementation of RETRIEVE Strategy from the paper :footcite:`killamsetty2020glister`  for supervised learning frameworks.
    RETRIEVE methods tries to solve the  bi-level optimization problem given below:

    .. math::
        \\overbrace{\\underset{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\operatorname{argmin\\hspace{0.7mm}}} L_V(\\underbrace{\\underset{\\theta}{\\operatorname{argmin\\hspace{0.7mm}}} L_T( \\theta, S)}_{inner-level}, {\\mathcal V})}^{outer-level}

    In the above equation, :math:`\\mathcal{U}` denotes the training set, :math:`\\mathcal{V}` denotes the validation set that guides the subset selection process, :math:`L_T` denotes the
    training loss, :math:`L_V` denotes the validation loss, :math:`S` denotes the data subset selected at each round,  and :math:`k` is the budget for the subset.

    Since, solving the complete inner-optimization is expensive, RETRIEVE adopts a online one-step meta approximation where we approximate the solution to inner problem
    by taking a single gradient step.

    The optimization problem after the approximation is as follows:

    .. math::
        \\overbrace{\\underset{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\operatorname{argmin\\hspace{0.7mm}}} L_V(\\underbrace{\\theta - \\eta \\nabla_{\\theta}L_T(\\theta, S)}_{inner-level}, {\\mathcal V})}^{outer-level}

    In the above equation, :math:`\\eta` denotes the step-size used for one-step gradient update.

    RETRIEVE-ONLINE also makes an additional approximation called Taylor-Series approximation to easily solve the outer problem using a greedy selection algorithm.
    The Taylor series approximation is as follows:

    .. math::
        L_V(\\theta - \\eta \\nabla_{\\theta}L_T(\\theta, S), {\\mathcal V}) \\approx L_V(\\theta) - \\eta {\\nabla_{\\theta}L_T(\\theta, S)}^T \\nabla_{\\theta}L_V(\\theta, {\\mathcal V})

    The Optimization problem after the Taylor series approximation is as follows:

    .. math::
        \\underset{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\operatorname{argmin\\hspace{0.7mm}}}L_V(\\theta - \\eta \\nabla_{\\theta}L_T(\\theta, S), {\\mathcal V}) \\approx L_V(\\theta) - \\eta {\\nabla_{\\theta}L_T(\\theta, S)}^T \\nabla_{\\theta}L_V(\\theta, {\\mathcal V})

    Taylor's series approximation reduces the time complexity by reducing the need of calculating the validation loss for each element during greedy selection step which
    means reducing the number of forward passes required.

    RETRIEVE-ONLINE is an adaptive subset selection algorithm that tries to select a subset every :math:`L` epochs and the parameter `L` can be set in the original training loop.

    Parameters
	----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    tea_model: class
        Teacher model architecture used for training
    ssl_alg: class
        SSL algorithm class
    loss: class
        Consistency loss function for unlabeled data with no reduction
    eta: float
        Learning rate. Step size for the one step gradient update
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    linear_layer: bool
        If True, we use the last fc layer weights and biases gradients
        If False, we use the last fc layer biases gradients
    selection_type: str
        Type of selection algorithm -
        - 'PerBatch' : PerBatch method is where RETRIEVE algorithm is applied on each minibatch data points.
        - 'PerClass' : PerClass method is where RETRIEVE algorithm is applied on each class data points seperately.
        - 'Supervised' : Supervised method is where RETRIEVE algorithm is applied on entire training data.
    greedy: str
        Type of greedy selection algorithm -
        - 'RGreedy' : RGreedy Selection method is a variant of naive greedy where we just perform r rounds of greedy selection by choosing k/r points in each round.
        - 'Stochastic' : Stochastic greedy selection method is based on the algorithm presented in this paper :footcite:`mirzasoleiman2014lazier`
        - 'Naive' : Normal naive greedy selection method that selects a single best element every step until the budget is fulfilled
    r : int, optional
        Number of greedy selection rounds when selection method is RGreedy (default: 15)
    valid: bool
        - If True, we select subset that maximizes the performance on the labeled set.
        - If False, we select subset that maximizes the performance on the unlabeled set.
    """

    def __init__(self, trainloader, valloader, model, tea_model, ssl_alg, loss,
                 eta, device, num_classes, linear_layer, selection_type, greedy, r=15, valid=True):
        """
        Constructor method
        """
        super().__init__(trainloader, valloader, model, tea_model, ssl_alg, num_classes, linear_layer, loss, device)
        self.eta = eta  # step size for the one step gradient update
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type
        self.r = r
        self.valid = valid
        self.greedy = greedy

    def _update_grads_val(self, grads_currX=None, first_init=False):
        """
        Update the gradient values

        Parameters
        ----------
        grad_currX: OrderedDict, optional
            Gradients of the current element (default: None)
        first_init: bool, optional
            Gradient initialization (default: False)
        """
        self.model.zero_grad()
        if self.selection_type == 'PerClass':
            valloader = self.pcvalloader
        else:
            valloader = self.valloader

        if self.selection_type == 'PerClass':
            trainloader = self.pctrainloader
        else:
            trainloader = self.trainloader
            
        embDim = self.model.get_embedding_dim()
        loss_name = self.loss.__class__.__name__
        if self.valid:
            if first_init:
                for batch_idx, (inputs, targets) in enumerate(valloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                    if loss_name == 'MeanSquared':
                        tmp_targets = torch.zeros(len(inputs), self.num_classes, device=self.device)
                        tmp_targets[torch.arange(len(inputs)), targets] = 1
                        targets = tmp_targets
                    if batch_idx == 0:
                        out, l1 = self.model(inputs, last=True, freeze=True)
                        if loss_name == 'MeanSquared':
                            temp_out = F.softmax(out, dim=1)
                            loss = F.mse_loss(temp_out, targets, reduction='none').sum()
                        else:
                            loss = F.cross_entropy(out, targets, reduction='none').sum()
                        l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                            l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                        self.init_out = out
                        self.init_l1 = l1                        
                        if self.selection_type == 'PerBatch':
                            l0_grads = l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                l1_grads = l1_grads.mean(dim=0).view(1, -1)                        
                        if loss_name == 'MeanSquared':
                            self.y_val = targets
                        else:
                            self.y_val = targets.view(-1, 1)
                    else:
                        out, l1 = self.model(inputs, last=True, freeze=True)
                        
                        if loss_name == 'MeanSquared':
                            temp_out = F.softmax(out, dim=1)
                            loss = F.mse_loss(temp_out, targets, reduction='none').sum()
                        else:
                            loss = F.cross_entropy(out, targets, reduction='none').sum()
                        
                        batch_l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                            batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                        
                        if self.selection_type == 'PerBatch':
                            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                        
                        l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                        if self.linear_layer:
                            l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
                        self.init_out = torch.cat((self.init_out, out), dim=0)
                        self.init_l1 = torch.cat((self.init_l1, l1), dim=0)
                        if loss_name == 'MeanSquared':
                            self.y_val = torch.cat((self.y_val, targets), dim=0)
                        else:
                            self.y_val = torch.cat((self.y_val, targets.view(-1, 1)), dim=0)
            elif grads_currX is not None:
                out_vec = self.init_out - (
                        self.eta * grads_currX[0][0:self.num_classes].view(1, -1).expand(self.init_out.shape[0], -1))
                if self.linear_layer:
                    out_vec = out_vec - (self.eta * torch.matmul(self.init_l1, grads_currX[0][self.num_classes:].view(
                        self.num_classes, -1).transpose(0, 1)))
                if loss_name == 'MeanSquared':
                    temp_out_vec = F.softmax(out_vec, dim=1)
                    loss = self.loss(temp_out_vec, self.y_val, torch.ones(len(temp_out_vec), device=self.device)).sum()
                else:
                    loss = self.loss(out_vec, self.y_val, torch.ones(len(out_vec), device=self.device)).sum()
                l0_grads = torch.autograd.grad(loss, out_vec)[0]
                if self.linear_layer:
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
                if self.selection_type == 'PerBatch':
                    b = int(l0_grads.shape[0]/self.valloader.batch_size)
                    l0_grads = torch.chunk(l0_grads, b, dim=0)
                    new_t = []
                    for i in range(len(l0_grads)):
                        new_t.append(torch.mean(l0_grads[i], dim=0).view(1, -1))
                    l0_grads = torch.cat(new_t, dim=0)
                    if self.linear_layer:
                        l1_grads = torch.chunk(l1_grads, b, dim=0)
                        new_t = []
                        for i in range(len(l1_grads)):
                            new_t.append(torch.mean(l1_grads[i], dim=0).view(1, -1))
                        l1_grads = torch.cat(new_t, dim=0)
            torch.cuda.empty_cache()
            if self.linear_layer:
                self.grads_val_curr = torch.mean(torch.cat((l0_grads, l1_grads), dim=1), dim=0).view(-1, 1)
            else:
                self.grads_val_curr = torch.mean(l0_grads, dim=0).view(-1, 1)
        else:
            if first_init:
                self.y_val = torch.cat(self.weak_targets, dim=0)
                for batch_idx, (ul_weak_aug, ul_strong_aug, _) in enumerate(trainloader):
                    ul_weak_aug, ul_strong_aug = ul_weak_aug.to(self.device), ul_strong_aug.to(self.device)
                    if batch_idx == 0:
                        out, l1 = self.model(ul_strong_aug, last=True, freeze=True)
                        if loss_name == 'MeanSquared':
                            temp_out = F.softmax(out, dim=1)
                            loss = self.loss(temp_out, self.weak_targets[batch_idx], self.weak_masks[batch_idx]).sum()
                        else:
                            loss = self.loss(out, self.weak_targets[batch_idx], self.weak_masks[batch_idx]).sum()
                        l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                            l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                        self.init_out = out
                        self.init_l1 = l1
                        if self.selection_type == 'PerBatch':
                            l0_grads = l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                l1_grads = l1_grads.mean(dim=0).view(1, -1)                        
                        
                    else:
                        out, l1 = self.model(ul_strong_aug, last=True, freeze=True)
                        if loss_name == 'MeanSquared':
                            temp_out = F.softmax(out, dim=1)
                            loss = self.loss(temp_out, self.weak_targets[batch_idx], self.weak_masks[batch_idx]).sum()
                        else:
                            loss = self.loss(out, self.weak_targets[batch_idx], self.weak_masks[batch_idx]).sum()
                        batch_l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                            batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                        
                        if self.selection_type == 'PerBatch':
                            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)

                        l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                        if self.linear_layer:
                            l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
                        self.init_out = torch.cat((self.init_out, out), dim=0)
                        self.init_l1 = torch.cat((self.init_l1, l1), dim=0)
            elif grads_currX is not None:
                out_vec = self.init_out - (
                            self.eta * grads_currX[0][0:self.num_classes].view(1, -1).expand(self.init_out.shape[0],-1))

                if self.linear_layer:
                    out_vec = out_vec - (self.eta * torch.matmul(self.init_l1, grads_currX[0][self.num_classes:].view(
                        self.num_classes, -1).transpose(0, 1)))

                if loss_name == 'MeanSquared':
                    temp_out_vec = F.softmax(out_vec, dim=1)
                    loss = self.loss(temp_out_vec, torch.cat(self.weak_targets, dim=0), torch.cat(self.weak_masks, dim=0)).sum()
                else:
                    loss = self.loss(out_vec, torch.cat(self.weak_targets, dim=0),
                                     torch.cat(self.weak_masks, dim=0)).sum()
                l0_grads = torch.autograd.grad(loss, out_vec)[0]
                if self.linear_layer:
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
                
                if self.selection_type == 'PerBatch':
                    b = int(l0_grads.shape[0]/self.valloader.batch_size)
                    l0_grads = torch.chunk(l0_grads, b, dim=0)
                    new_t = []
                    for i in range(len(l0_grads)):
                        new_t.append(torch.mean(l0_grads[i], dim=0).view(1, -1))
                    l0_grads = torch.cat(new_t, dim=0)
                    if self.linear_layer:
                        l1_grads = torch.chunk(l1_grads, b, dim=0)
                        new_t = []
                        for i in range(len(l1_grads)):
                            new_t.append(torch.mean(l1_grads[i], dim=0).view(1, -1))
                        l1_grads = torch.cat(new_t, dim=0)
            torch.cuda.empty_cache()
            if self.linear_layer:
                self.grads_val_curr = torch.mean(torch.cat((l0_grads, l1_grads), dim=1), dim=0).view(-1, 1)
            else:
                self.grads_val_curr = torch.mean(l0_grads, dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads):
        """
        Evaluate gradients

        Parameters
        ----------
        grads: Tensor
            Gradients

        Returns
        ----------
        gains: Tensor
            Matrix product of two tensors
        """

        grads_val = self.grads_val_curr
        with torch.no_grad():
            gains = torch.matmul(grads, grads_val)
        return gains

    def _update_gradients_subset(self, grads_X, element):
        """
        Update gradients of set X + element (basically adding element to X)
        Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!

        Parameters
        ----------
        grads_X: list
            Gradients
        element: int
            Element that need to be added to the gradients
        """
        # if isinstance(element, list):
        grads_X += self.grads_per_elem[element].sum(dim=0)

    def greedy_algo(self, budget):
        greedySet = list()
        N = self.grads_per_elem.shape[0]
        remainSet = list(range(N))
        t_ng_start = time.time()  # naive greedy start time
        numSelected = 0
        if self.greedy == 'RGreedy':
            # subset_size = int((len(self.grads_per_elem) / r))
            selection_size = int(budget / self.r)
            while (numSelected < budget):
                # Try Using a List comprehension here!
                rem_grads = self.grads_per_elem[remainSet]
                gains = self.eval_taylor_modular(rem_grads)
                # Update the greedy set and remaining set
                sorted_gains, indices = torch.sort(gains.view(-1), descending=True)
                selected_indices = [remainSet[index.item()] for index in indices[0:selection_size]]
                greedySet.extend(selected_indices)
                [remainSet.remove(idx) for idx in selected_indices]
                if numSelected == 0:
                    grads_curr = self.grads_per_elem[selected_indices].sum(dim=0).view(1, -1)
                else:  # If 1st selection, then just set it to bestId grads
                    self._update_gradients_subset(grads_curr, selected_indices)
                # Update the grads_val_current using current greedySet grads
                self._update_grads_val(grads_curr)
                numSelected += selection_size
            print("R greedy RETRIEVE total time:", time.time() - t_ng_start)

        # Stochastic Greedy Selection Algorithm
        elif self.greedy == 'Stochastic':
            subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
            while (numSelected < budget):
                # Try Using a List comprehension here!
                subset_selected = random.sample(remainSet, k=subset_size)
                rem_grads = self.grads_per_elem[subset_selected]
                gains = self.eval_taylor_modular(rem_grads)
                # Update the greedy set and remaining set
                _, indices = torch.sort(gains.view(-1), descending=True)
                bestId = [subset_selected[indices[0].item()]]
                greedySet.append(bestId[0])
                remainSet.remove(bestId[0])
                numSelected += 1
                # Update info in grads_currX using element=bestId
                if numSelected > 1:
                    self._update_gradients_subset(grads_curr, bestId)
                else:  # If 1st selection, then just set it to bestId grads
                    grads_curr = self.grads_per_elem[bestId].view(1, -1)  # Making it a list so that is mutable!
                # Update the grads_val_current using current greedySet grads
                self._update_grads_val(grads_curr)
            print("Stochastic Greedy RETRIEVE total time:", time.time() - t_ng_start)

        elif self.greedy == 'Naive':
            while (numSelected < budget):
                # Try Using a List comprehension here!
                rem_grads = self.grads_per_elem[remainSet]
                gains = self.eval_taylor_modular(rem_grads)
                # Update the greedy set and remaining set
                # _, maxid = torch.max(gains, dim=0)
                _, indices = torch.sort(gains.view(-1), descending=True)
                bestId = [remainSet[indices[0].item()]]
                greedySet.append(bestId[0])
                remainSet.remove(bestId[0])
                numSelected += 1
                # Update info in grads_currX using element=bestId
                if numSelected == 1:
                    grads_curr = self.grads_per_elem[bestId[0]].view(1, -1)
                else:  # If 1st selection, then just set it to bestId grads
                    self._update_gradients_subset(grads_curr, bestId)
                # Update the grads_val_current using current greedySet grads
                self._update_grads_val(grads_curr)
            print("Naive Greedy RETRIEVE total time:", time.time() - t_ng_start)
        return list(greedySet), [1] * budget

    def select(self, budget, model_params, tea_model_params):
        """
        Apply naive greedy method for data selection

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
        greedySet: list
            List containing indices of the best datapoints,
        budget: Tensor
            Tensor containing gradients of datapoints present in greedySet
        """
        glister_start_time = time.time() # naive greedy start time
        self.update_model(model_params, tea_model_params)
        if self.selection_type == 'PerClass':
            self.get_labels(valid=True)
            idxs = []
            gammas = []
            for i in range(self.num_classes):
                trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()
                trn_data_sub = Subset(self.trainloader.dataset, trn_subset_idx)
                self.pctrainloader = DataLoader(trn_data_sub, batch_size=self.trainloader.batch_size,
                                                shuffle=False, pin_memory=True)
                
                val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
                val_data_sub = Subset(self.valloader.dataset, val_subset_idx)
                self.pcvalloader = DataLoader(val_data_sub, batch_size=self.trainloader.batch_size,
                                            shuffle=False, pin_memory=True)
                if self.valid:
                    self.compute_gradients(store_t=False, perClass=True)
                else:
                    self.compute_gradients(store_t=True, perClass=True)
                
                self._update_grads_val(first_init=True)
                idxs_temp, gammas_temp = self.greedy_algo(math.ceil(budget * len(trn_subset_idx) / self.N_trn))
                idxs.extend(list(np.array(trn_subset_idx)[idxs_temp]))
                gammas.extend(gammas_temp)
        elif self.selection_type == 'PerBatch':
            idxs = []
            gammas = []
            if self.valid:
                self.compute_gradients(store_t=False, perBatch=True)
            else:
                self.compute_gradients(store_t=True, perBatch=True)
            self._update_grads_val(first_init=True)
            idxs_temp, gammas_temp = self.greedy_algo(math.ceil(budget/self.trainloader.batch_size))
            batch_wise_indices = list(self.trainloader.batch_sampler)
            for i in range(len(idxs_temp)):
                tmp = batch_wise_indices[idxs_temp[i]]
                idxs.extend(tmp)
                gammas.extend([gammas_temp[i]] * len(tmp))
        else:
            if self.valid:
                self.compute_gradients(store_t=False)
            else:
                self.compute_gradients(store_t=True)
            self._update_grads_val(first_init=True)
            idxs, gammas = self.greedy_algo(budget)
        glister_end_time = time.time()
        print("RETRIEVE algorithm Subset Selection time is: ", glister_end_time - glister_start_time)
        return idxs, torch.FloatTensor(gammas)

