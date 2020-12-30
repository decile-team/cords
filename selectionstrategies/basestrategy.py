import time
import copy
import datetime
import os
import subprocess
import sys
import math
import numpy as np
import apricot
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split, SequentialSampler, BatchSampler, RandomSampler
from torch.autograd import Variable
import torch.nn.functional as F


class BaseStrategy(object):
    '''
    Parameters
    ----------
    trainloader : Tensor
        The training dataset
    valloader : Tensor
        The validation dataset
    model : class
        The type of model used for training
    loss_criterion : class
        The type of loss criterion
    loss_nored : class
        Specifies that no reduction will be applied to  the output
    eta : float
        Learning rate. step size for the one step gradient update
    device : str
        The device being utilized - 'cpu' | 'cuda'
    num_channels : int
        The number of channels
    num_classes : int
        The number of target classes in the dataset
    batch_size : int
        The batch size of the training data
    bud : int
        The number of data points to be selected
    if_convex : bool
        True | False
    N_trn : int
        The number of training instances in the dataset
    grads_per_elem : Tensor
        Initialize gradients of each element to None at the beginning of the training
        Tensor containing gradients of each element
    theta_init : OrderedDict
        Initialize theta to None at the beginning of the training
        OrderedDict containing tensors of weights and biases
    numSelected : int
    '''

    def __init__(self, trainloader, valloader, model, loss_criterion,
                 loss_nored, eta, device, num_channels, num_classes, batch_size, bud, if_convex):

        self.trainloader = trainloader  # assume its a sequential loader.
        self.valloader = valloader  # assume its a sequential loader.
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        self.device = device
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.bud = bud
        self.if_convex = if_convex
        self.N_trn = len(trainloader)
        self.grads_per_elem = None
        self.theta_init = None
        self.numSelected = 0


    def _compute_per_element_grads(self, theta_init):
        '''
        Computes the gradient of each element
        '''
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        grads_vec = [0 for i in range(self.N_trn)]
        counter = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            scores = self.model(inputs)
            losses = self.loss_nored(scores, targets)
            for i, loss_i in enumerate(losses):
                grads_vec[i + counter] = torch.autograd.grad(loss_i,
                                                             self.model.parameters(), retain_graph=True)
            counter += len(targets)
        self.grads_per_elem = grads_vec

    
    def _simple_eval(self, grads_X, theta_init):
        '''
        Computes the validation loss using the vector of gradient sums in set X.
        '''

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad():
            # perform one-step update
            for i, param in enumerate(self.model.parameters()):
                param.data.sub_(self.eta * grads_X[i])
            val_loss = 0
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                scores = self.model(inputs)
                val_loss += self.loss(scores, targets)
        return -1.0 * val_loss.item()


    def eval(self, grads_X, grads_elem, theta_init):
        '''
        Computes the Validation Loss using the subset: X + elem by utilizing the
        gradient of model parameters.
        '''

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad():  # perform one-step update
            for i, param in enumerate(self.model.parameters()):
                param.data.sub_(self.eta * (grads_X[i] + grads_elem[i]))
            val_loss = 0
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                scores = self.model(inputs)
                val_loss += self.loss(scores, targets)
        return -1.0 * val_loss.item()


    def _update_gradients_subset(self, grads_X, element):
        '''
        Updates gradients of set X + element (basically adding element to X)
        Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
        '''

        grads_e = self.grads_per_elem[element]
        for i, _ in enumerate(self.model.parameters()):
            grads_X[i] += grads_e[i]


    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init):
        '''
        Apply naive greedy method for data selection
        '''

        self._compute_per_element_grads(theta_init)
        print("Computed train set gradients")
        # Dont need the trainloader here!! Same as full batch version!
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        remainSet = set(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        while (numSelected < budget):
            bestGain = -np.inf  # curr Val for gain
            bestId = -1
            t_one_elem = time.time()
            for i in remainSet:
                grads_i = self.grads_per_elem[i]
                ## If no elements selected, use the self._simple_eval to get validation loss
                val_i = self.eval(grads_currX, grads_i, theta_init) if numSelected > 0 else self._simple_eval(grads_i,
                                                                                                              theta_init)
                if val_i > bestGain:
                    bestGain = val_i
                    bestId = i
            # Update the greedy set and remaining set
            greedySet.add(bestId)
            remainSet.remove(bestId)
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:
                # If 1st selection, then just set it to bestId grads
                # Making it a list so that is mutable!
                grads_currX = list(self.grads_per_elem[bestId])
            if numSelected % 200 == 0:
                # Printing the Validation Loss
                # Also print the selection time for 1 element
                print("numSelected", numSelected, "Time for 1selc:", time.time() - t_one_elem, "ValLoss:",
                      -1 * bestGain)
            numSelected += 1
        print("Naive greedy time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


    def distance(self, x, y, exp=2):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist


    def compute_score(self, model):
        self.N = 0
        g_is = []
        with torch.no_grad():
            for i, data_i in enumerate(self.trainloader, 0):
                inputs_i, target_i = data_i
                inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
                self.N += inputs_i.size()[0]

                if not self.if_convex:
                    scores_i = F.softmax(model(inputs_i), dim=1)
                    y_i = torch.zeros(target_i.size(0), scores_i.size(1)).to(self.device)
                    y_i[range(y_i.shape[0]), target_i] = 1
                    g_is.append(scores_i - y_i)
                else:
                    g_is.append(inputs_i)

            self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                # print(i,end=",")
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False

                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j)
        self.dist_mat = self.dist_mat.cpu().numpy()


    def compute_gamma(self, idxs):
        gamma = [0 for i in range(len(idxs))]
        best = self.dist_mat[idxs]  # .to(self.device)
        rep = np.argmax(best, axis=0)
        for i in rep:
            gamma[i] += 1
        return gamma


    def get_similarity_kernel(self):
        for i, data_i in enumerate(self.trainloader, 0):
            if i == 0:
                _, targets = data_i
            else:
                _, target_i = data_i
                targets = torch.cat((targets, target_i), dim=0)
        kernel = np.zeros((targets.shape[0], targets.shape[0]))
        targets = targets.cpu().numpy()
        for target in np.unique(targets):
            x = np.where(targets == target)[0]
            # prod = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
            for i in x:
                kernel[i, x] = 1
        return kernel


    def lazy_greedy_max(self, budget, model):
        self.compute_score(model)
        kernel = self.get_similarity_kernel()
        fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                          n_samples=budget)
        self.dist_mat = self.dist_mat * kernel
        sim_sub = fl.fit_transform(self.dist_mat)
        greedyList = list(np.argmax(sim_sub, axis=1))
        gamma = self.compute_gamma(greedyList)
        return greedyList, gamma


    def model_eval_loss(data_loader, model, criterion):
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss
