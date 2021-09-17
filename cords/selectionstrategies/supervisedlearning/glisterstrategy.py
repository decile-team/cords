import math
import random
import time
import torch
import torch.nn.functional as F
from .dataselectionstrategy import DataSelectionStrategy


class GLISTERStrategy(DataSelectionStrategy):
    """
    Implementation of GLISTER-ONLINE Strategy from the paper :footcite:`killamsetty2020glister`  for supervised learning frameworks.
    GLISTER-ONLINE methods tries to solve the  bi-level optimization problem given below:

    .. math::
        \\overbrace{\\underset{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\operatorname{argmin\\hspace{0.7mm}}} L_V(\\underbrace{\\underset{\\theta}{\\operatorname{argmin\\hspace{0.7mm}}} L_T( \\theta, S)}_{inner-level}, {\\mathcal V})}^{outer-level}

    In the above equation, :math:`\\mathcal{U}` denotes the training set, :math:`\\mathcal{V}` denotes the validation set that guides the subset selection process, :math:`L_T` denotes the
    training loss, :math:`L_V` denotes the validation loss, :math:`S` denotes the data subset selected at each round,  and :math:`k` is the budget for the subset.

    Since, solving the complete inner-optimization is expensive, GLISTER-ONLINE adopts a online one-step meta approximation where we approximate the solution to inner problem
    by taking a single gradient step.

    The optimization problem after the approximation is as follows:

    .. math::
        \\overbrace{\\underset{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\operatorname{argmin\\hspace{0.7mm}}} L_V(\\underbrace{\\theta - \\eta \\nabla_{\\theta}L_T(\\theta, S)}_{inner-level}, {\\mathcal V})}^{outer-level}

    In the above equation, :math:`\\eta` denotes the step-size used for one-step gradient update.

    GLISTER-ONLINE also makes an additional approximation called Taylor-Series approximation to easily solve the outer problem using a greedy selection algorithm.
    The Taylor series approximation is as follows:

    .. math::
        L_V(\\theta - \\eta \\nabla_{\\theta}L_T(\\theta, S), {\\mathcal V}) \\approx L_V(\\theta) - \\eta {\\nabla_{\\theta}L_T(\\theta, S)}^T \\nabla_{\\theta}L_V(\\theta, {\\mathcal V})

    The Optimization problem after the Taylor series approximation is as follows:

    .. math::
        \\underset{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\operatorname{argmin\\hspace{0.7mm}}}L_V(\\theta - \\eta \\nabla_{\\theta}L_T(\\theta, S), {\\mathcal V}) \\approx L_V(\\theta) - \\eta {\\nabla_{\\theta}L_T(\\theta, S)}^T \\nabla_{\\theta}L_V(\\theta, {\\mathcal V})

    Taylor's series approximation reduces the time complexity by reducing the need of calculating the validation loss for each element during greedy selection step which
    means reducing the number of forward passes required.

    GLISTER-ONLINE is an adaptive subset selection algorithm that tries to select a subset every :math:`L` epochs and the parameter `L` can be set in the original training loop.

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
        If True, we use the last fc layer weights and biases gradients
        If False, we use the last fc layer biases gradients
    selection_type: str
        Type of selection -
        - 'RGreedy' : RGreedy Selection method is a variant of naive greedy where we just perform r rounds of greedy selection by choosing k/r points in each round.
        - 'Stochastic' : Stochastic greedy selection method is based on the algorithm presented in this paper :footcite:`mirzasoleiman2014lazier`
        - 'Naive' : Normal naive greedy selection method that selects a single best element every step until the budget is fulfilled
    r : int, optional
        Number of greedy selection rounds when selection method is RGreedy (default: 15)
    """

    def __init__(self, trainloader, valloader, model, loss_type,
                 eta, device, num_classes, linear_layer, selection_type, r=15):
        """
        Constructor method
        """

        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss_type, device)
        self.eta = eta  # step size for the one step gradient update
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type
        self.r = r

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
        embDim = self.model.get_embedding_dim()

        if first_init:
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    self.init_out = out
                    self.init_l1 = l1
                    self.y_val = targets.view(-1, 1)
                else:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

                    self.init_out = torch.cat((self.init_out, out), dim=0)
                    self.init_l1 = torch.cat((self.init_l1, l1), dim=0)
                    self.y_val = torch.cat((self.y_val, targets.view(-1, 1)), dim=0)

        elif grads_currX is not None:
            out_vec = self.init_out - (
                        self.eta * grads_currX[0][0:self.num_classes].view(1, -1).expand(self.init_out.shape[0],-1))

            if self.linear_layer:
                out_vec = out_vec - (self.eta * torch.matmul(self.init_l1, grads_currX[0][self.num_classes:].view(
                    self.num_classes, -1).transpose(0, 1)))

            loss = self.loss(out_vec, self.y_val.view(-1)).sum()
            l0_grads = torch.autograd.grad(loss, out_vec)[0]
            if self.linear_layer:
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
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

    def select(self, budget, model_params):
        """
        Apply naive greedy method for data selection

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
        self.update_model(model_params)
        self.compute_gradients()
        t_ng_start = time.time()  # naive greedy start time
        self._update_grads_val(first_init=True)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        greedySet = list()
        remainSet = list(range(self.N_trn))
        # RModular Greedy Selection Algorithm
        if self.selection_type == 'RGreedy':
            # subset_size = int((len(self.grads_per_elem) / r))
            selection_size = int(budget / self.r)
            while (self.numSelected < budget):
                # Try Using a List comprehension here!
                rem_grads = self.grads_per_elem[remainSet]
                gains = self.eval_taylor_modular(rem_grads)
                # Update the greedy set and remaining set
                sorted_gains, indices = torch.sort(gains.view(-1), descending=True)
                selected_indices = [remainSet[index.item()] for index in indices[0:selection_size]]
                greedySet.extend(selected_indices)
                [remainSet.remove(idx) for idx in selected_indices]
                if self.numSelected == 0:
                    grads_currX = self.grads_per_elem[selected_indices].sum(dim=0).view(1, -1)
                else:  # If 1st selection, then just set it to bestId grads
                    self._update_gradients_subset(grads_currX, selected_indices)
                # Update the grads_val_current using current greedySet grads
                self._update_grads_val(grads_currX)
                self.numSelected += selection_size
            print("R greedy GLISTER total time:", time.time() - t_ng_start)

        # Stochastic Greedy Selection Algorithm
        elif self.selection_type == 'Stochastic':
            subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
            while (self.numSelected < budget):
                # Try Using a List comprehension here!
                subset_selected = random.sample(remainSet, k=subset_size)
                rem_grads = self.grads_per_elem[subset_selected]
                gains = self.eval_taylor_modular(rem_grads)
                # Update the greedy set and remaining set
                _, indices = torch.sort(gains.view(-1), descending=True)
                bestId = [subset_selected[indices[0].item()]]
                greedySet.append(bestId[0])
                remainSet.remove(bestId[0])
                self.numSelected += 1
                # Update info in grads_currX using element=bestId
                if self.numSelected > 1:
                    self._update_gradients_subset(grads_currX, bestId)
                else:  # If 1st selection, then just set it to bestId grads
                    grads_currX = self.grads_per_elem[bestId].view(1, -1) 
                # Making it a list so that is mutable!
                # Update the grads_val_current using current greedySet grads
                self._update_grads_val(grads_currX)
            print("Stochastic Greedy GLISTER total time:", time.time() - t_ng_start)

        elif self.selection_type == 'Naive':
            while (self.numSelected < budget):
                # Try Using a List comprehension here!
                rem_grads = self.grads_per_elem[remainSet]
                gains = self.eval_taylor_modular(rem_grads)
                # Update the greedy set and remaining set
                # _, maxid = torch.max(gains, dim=0)
                _, indices = torch.sort(gains.view(-1), descending=True)
                bestId = [remainSet[indices[0].item()]]
                greedySet.append(bestId[0])
                remainSet.remove(bestId[0])
                self.numSelected += 1
                # Update info in grads_currX using element=bestId
                if self.numSelected == 1:
                    grads_currX = self.grads_per_elem[bestId[0]].view(1, -1)
                else:  # If 1st selection, then just set it to bestId grads
                    self._update_gradients_subset(grads_currX, bestId)
                # Update the grads_val_current using current greedySet grads
                self._update_grads_val(grads_currX)
            print("Naive Greedy GLISTER total time:", time.time() - t_ng_start)
        return list(greedySet), torch.ones(budget)
