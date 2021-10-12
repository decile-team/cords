import numpy as np
import torch
import torch.nn.functional as F


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

    def compute_gradients(self, valid=False, perBatch=False, perClass=False):
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
        perBatch: bool
            if True, the function computes the gradients of each mini-batch
        perClass: bool
            if True, the function computes the gradients using perclass dataloaders
        """
        if (perBatch and perClass):
            raise ValueError("batch and perClass are mutually exclusive. Only one of them can be true at a time")

        embDim = self.model.get_embedding_dim()
        if perClass:
            trainloader = self.pctrainloader
            if valid:
                valloader = self.pcvalloader
        else:
            trainloader = self.trainloader
            if valid:
                valloader = self.valloader

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if batch_idx == 0:
                out, l1 = self.model(inputs, last=True, freeze=True)
                loss = self.loss(out, targets).sum()
                l0_grads = torch.autograd.grad(loss, out)[0]
                if self.linear_layer:
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                if perBatch:
                    l0_grads = l0_grads.mean(dim=0).view(1, -1)
                    if self.linear_layer:
                        l1_grads = l1_grads.mean(dim=0).view(1, -1)
            else:
                out, l1 = self.model(inputs, last=True, freeze=True)
                loss = self.loss(out, targets).sum()
                batch_l0_grads = torch.autograd.grad(loss, out)[0]
                if self.linear_layer:
                    batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                    batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

                if perBatch:
                    batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                    if self.linear_layer:
                        batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                if self.linear_layer:
                    l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

        torch.cuda.empty_cache()

        if self.linear_layer:
            self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
        else:
            self.grads_per_elem = l0_grads

        if valid:
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if perBatch:
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)
                else:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

                    if perBatch:
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
            torch.cuda.empty_cache()
            if self.linear_layer:
                self.val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
            else:
                self.val_grads_per_elem = l0_grads

    def update_model(self, model_params):
        """
        Update the models parameters
        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        """
        self.model.load_state_dict(model_params)
