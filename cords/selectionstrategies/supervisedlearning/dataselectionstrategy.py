import logging
from abc import abstractmethod

import torch


class DataSelectionStrategy(object):
    """
    Implementation of Data Selection Strategy class which serves as base class for other
    dataselectionstrategies for general learning frameworks.
    Parameters
        ----------
        trainloader: class
            Loading the training data using pytorch dataloader
        valloader: class
            Loading the validation data using pytorch dataloader
        model: class
            Model architecture used for training
        num_classes: int
            Number of target classes in the dataset
        linear_layer: bool
            If True, we use the last fc layer weights and biases gradients
            If False, we use the last fc layer biases gradients
        loss: class
            PyTorch Loss function
    """

    def __init__(self, trainloader, valloader, model, num_classes, linear_layer, loss, device):
        """
        Constructer method
        """
        self.trainloader = trainloader  # assume its a sequential loader.
        self.valloader = valloader
        self.model = model
        self.N_trn = len(trainloader.sampler)
        self.N_val = len(valloader.sampler)
        self.grads_per_elem = None
        self.val_grads_per_elem = None
        self.numSelected = 0
        self.linear_layer = linear_layer
        self.num_classes = num_classes
        self.trn_lbls = None
        self.val_lbls = None
        self.loss = loss
        self.device = device
        self.model_embedding_dim = self.model.get_embedding_dim()

    def select(self, budget, model_params):
        pass

    def get_labels(self, valid=False):
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                self.trn_lbls = targets.view(-1, 1)
            else:
                self.trn_lbls = torch.cat((self.trn_lbls, targets.view(-1, 1)), dim=0)
        self.trn_lbls = self.trn_lbls.view(-1)

        if valid:
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                if batch_idx == 0:
                    self.val_lbls = targets.view(-1, 1)
                else:
                    self.val_lbls = torch.cat((self.val_lbls, targets.view(-1, 1)), dim=0)
            self.val_lbls = self.val_lbls.view(-1)

    @abstractmethod
    def approximators_gradient_handler(self, approximators, loss, batch):
        raise Exception('Not implemented. ')

    def compute_gradients(self, valid=False, batch=False, perClass=False):
        """
        Computes the gradient of each element.

        Here, the gradients are computed in a closed form using CrossEntropyLoss with reduction set to 'none'.
        This is done by calculating the gradients in last layer through addition of softmax layer.

        Using different loss functions, the way we calculate the gradients will change.

        For LogisticLoss we measure the Mean Absolute Error(MAE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:

        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left| x_n - y_n \\right|,

        where :math:`N` is the batch size.


        For MSELoss, we measure the Mean Square Error(MSE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:

        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left( x_n - y_n \\right)^2,

        where :math:`N` is the batch size.
        Parameters
        ----------
        valid: bool
            if True, the function also computes the validation gradients
        batch: bool
            if True, the function computes the gradients of each mini-batch
        perClass: bool
            if True, the function computes the gradients using perclass dataloaders
        """
        grads = []
        train_loader = self.pctrainloader if perClass else self.trainloader

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            out, approximator = self.model(inputs, last=True, freeze=True)
            # breakpoint()
            loss = self.loss(out, targets).sum()
            grads += [self.approximators_gradient_handler(approximator, loss, batch)]
        torch.cuda.empty_cache()
        self.grads_per_elem = torch.cat(grads, dim=0)

        # valid
        val_grads = []
        if valid:
            for batch_idx, (inputs, targets) in enumerate(self.pcvalloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                out, approximator = self.model(inputs, last=True, freeze=True)
                loss = self.loss(out, targets).sum()
                val_grads += [self.approximators_gradient_handler(approximator, loss, batch)]
            torch.cuda.empty_cache()
            self.val_grads_per_elem = torch.cat(val_grads, dim=0)

    def update_val_gradients(self, grads_currX=None, first_init=False, batch=True):
        """
        Update the gradient values

        Parameters
        ----------
        grad_currX: OrderedDict, optional
            Gradients of the current element (default: None)
        first_init: bool, optional
            Gradient initialization (default: False)
        """

        # if self.verbose:
        #     logging.info('Updating validation set gradient. ')

        self.model.zero_grad()
        embDim = self.model.get_embedding_dim()

        if first_init:
            ################################################################
            # TODO-2: l1 and approximator to be refactored after TODO-1
            # val_grads, y_val_arr, out_arr, approximator_arr = [], [], [], []
            val_grads, y_val_arr, out_arr, l1_arr = [], [], [], []
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                out, approximator = self.model(inputs, last=True, freeze=True)
                loss = self.loss(out, targets).sum()
                val_grads += [self.approximators_gradient_handler(approximator, loss, batch)]
                out_arr += [out]
                y_val_arr += [targets.view(-1, 1)]
                l1_arr += [approximator[1]]
                self.out = torch.cat(out_arr, dim=0)
                self.l1 = torch.cat(l1_arr, dim=0)
                self.y_val = torch.cat(y_val_arr, dim=0)
                self.val_grads_per_elem = torch.cat(val_grads, dim=0)
            ################################################################
        else:
            ################################################################
            # TODO-1: integrate in approximators_gradient_handler
            if grads_currX is None:
                raise Exception('Not first time initialization must have a grads_currX! ')
            out_vec = self.out - (
                        self.eta * grads_currX[0][0:self.num_classes].view(1, -1).expand(self.out.shape[0], -1))

            if self.linear_layer:
                out_vec = out_vec - (self.eta * torch.matmul(self.l1, grads_currX[0][self.num_classes:].view(
                    self.num_classes, -1).transpose(0, 1)))

            loss = self.loss(out_vec, self.y_val.view(-1)).sum()
            l0_grads = torch.autograd.grad(loss, out_vec)[0]
            if self.linear_layer:
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.l1.repeat(1, self.num_classes)
                self.val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
            else:
                self.val_grads_per_elem = l0_grads
            ################################################################
        torch.cuda.empty_cache()

    def update_model(self, model_params):
        """
        Update the models parameters

        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        """
        self.model.load_state_dict(model_params)
