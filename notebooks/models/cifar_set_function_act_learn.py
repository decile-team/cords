import numpy as np
import time
import torch
import math
import torch.nn.functional as F
from torch.utils.data import random_split, SequentialSampler, BatchSampler
from queue import PriorityQueue
from torch import random

class SetFunctionLoader_2(object):
    def __init__(self, trainset, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes, batch_size):
        self.trainset = trainset  # assume its a sequential loader.
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = len(trainset)
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.batch_size = batch_size

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(self.remainList), self.batch_size, drop_last=False))][0])
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [self.trainset[x][0].view(-1, 3, self.trainset[x][0].shape[1], self.trainset[x][0].shape[2]) for x in
                 batch_idx], dim=0).type(torch.float)
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if cnt == 0:
                with torch.no_grad():
                    data = F.softmax(self.model(inputs), dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                cnt = cnt + 1
            else:
                cnt = cnt + 1
                with torch.no_grad():
                    data = torch.cat((data, F.softmax(self.model(inputs), dim=1)), dim=0)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = torch.cat((outputs, tmp_tensor), dim=0)
        grads_vec = data - outputs
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        self.N_trn = len(grads_vec)
        self.grads_per_elem = grads_vec

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            with torch.no_grad():
                scores = F.softmax(self.model(self.x_val), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(self.eta * grads_currX)
                scores = F.softmax(self.model(self.x_val), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        self.grads_val_curr = grads.mean(dim=0)  # reset parm.grads to zero!

    def eval_taylor(self, grads_elem, theta_init):
        grads_val = self.grads_val_curr
        dot_prod = 0
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            params = [param for param in self.model.parameters()]
            dot_prod += torch.sum(grads_val[0] * (params[-1].data - self.eta * grads_elem[0]))
        return dot_prod.data

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            grads_tensor = torch.cat(grads, dim=0)
            param_update = self.eta * grads_tensor
            gains = torch.matmul(param_update, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X += grads_e

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget,remainList,theta_init):
        self.remainList = remainList
        start_time = time.time()
        self._compute_per_element_grads(theta_init)
        end_time = time.time()
        print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = [self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0]) for x in subset_selected]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains)]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if numSelected % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem)
            numSelected += 1
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


class PriorSetFunctionLoader_2(object):

    def __init__(self, trainset, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes, batch_size):
        self.trainset = trainset  # assume its a sequential loader.
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = len(trainset)
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.batch_size = batch_size

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(np.arange(self.N_trn)), self.batch_size, drop_last=False))][0])
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [self.trainset[x][0].view(-1, 3, self.trainset[x][0].shape[1], self.trainset[x][0].shape[2]) for x in
                 batch_idx], dim=0).type(torch.float)
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if cnt == 0:
                with torch.no_grad():
                    data = F.softmax(self.model(inputs), dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                cnt = cnt + 1
            else:
                cnt = cnt + 1
                with torch.no_grad():
                    data = torch.cat((data, F.softmax(self.model(inputs), dim=1)), dim=0)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = torch.cat((outputs, tmp_tensor), dim=0)
        grads_vec = data - outputs
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = grads_vec

    def _update_grads_val(self, theta_init, label, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        label_indices = torch.where(self.y_val == label)[0]
        if first_init:
            with torch.no_grad():
                scores = F.softmax(self.model(self.x_val[label_indices]), dim=1)
                one_hot_label = torch.zeros(len(self.y_val[label_indices]), self.num_classes).to(self.device)
                one_hot_label[:, label] = 1
                grads = scores - one_hot_label
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(self.eta * grads_currX)
                scores = F.softmax(self.model(self.x_val[label_indices]), dim=1)
                one_hot_label = torch.zeros(len(self.y_val[label_indices]), self.num_classes).to(self.device)
                one_hot_label[:, label] = 1
                grads = scores - one_hot_label
        self.grads_val_curr = grads.mean(dim=0)  # reset parm.grads to zero!

    def eval_taylor(self, grads_elem, theta_init):
        grads_val = self.grads_val_curr
        dot_prod = 0
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            params = [param for param in self.model.parameters()]
            dot_prod += torch.sum(grads_val[0] * (params[-1].data - self.eta * grads_elem[0]))
        return dot_prod.data

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            grads_tensor = torch.cat(grads, dim=0)
            param_update = self.eta * grads_tensor
            gains = torch.matmul(param_update, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X += grads_e

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init):
        start_time = time.time()
        self._compute_per_element_grads(theta_init)
        end_time = time.time()
        print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        t_ng_start = time.time()  # naive greedy start time
        # subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        per_class_budget = int(budget / self.num_classes)
        totalSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        for i in range(self.num_classes):
            if totalSelected == 0:
                self._update_grads_val(theta_init, label=i, first_init=True)
            else:
                self._update_grads_val(theta_init, i, grads_currX)
            numSelected = 0
            greedySet = list()
            remainSet = [x.item() for x in
                         torch.where(torch.tensor(self.trainset.dataset.targets)[self.trainset.indices] == i)[0]]
            subset_size = int((len(remainSet) / per_class_budget) * math.log(100))
            while (numSelected < per_class_budget):
                # Try Using a List comprehension here!
                t_one_elem = time.time()
                subset_selected = remainSet  # list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
                rem_grads = [self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0]) for x in subset_selected]
                gains = self.eval_taylor_modular(rem_grads, theta_init)
                # Update the greedy set and remaining set
                bestId = subset_selected[torch.argmax(gains)]
                greedySet.append(bestId)
                remainSet.remove(bestId)
                # Update info in grads_currX using element=bestId
                if totalSelected > 0:
                    self._update_gradients_subset(grads_currX, bestId)
                else:  # If 1st selection, then just set it to bestId grads
                    grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
                # Update the grads_val_current using current greedySet grads
                self._update_grads_val(theta_init, i, grads_currX)
                if numSelected % 1000 == 0:
                    # Printing bestGain and Selection time for 1 element.
                    print("numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem)
                numSelected += 1
                totalSelected += 1
            print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


class WeightedSetFunctionLoader(object):
    def __init__(self, trainset, x_val, y_val, facloc_size, lam, model, loss_criterion,
                 loss_nored, eta, device, num_classes, batch_size):
        self.trainset = trainset  # assume its a sequential loader.
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.facloc_size = facloc_size
        self.lam = lam
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = len(trainset)
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.batch_size = batch_size

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(self.remainList), self.batch_size, drop_last=False))][0])
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [self.trainset[x][0].view(-1, 3, self.trainset[x][0].shape[1], self.trainset[x][0].shape[2]) for x in
                 batch_idx], dim=0).type(torch.float)
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if cnt == 0:
                with torch.no_grad():
                    data = F.softmax(self.model(inputs), dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                cnt = cnt + 1
            else:
                cnt = cnt + 1
                with torch.no_grad():
                    data = torch.cat((data, F.softmax(self.model(inputs), dim=1)), dim=0)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = torch.cat((outputs, tmp_tensor), dim=0)
        grads_vec = data - outputs
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = grads_vec

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            with torch.no_grad():
                for i in range(10):
                    batch_scores = F.softmax(self.model(
                        self.x_val[(i) * int((len(self.x_val)) / 10): (i + 1) * int((len(self.x_val)) / 10)]), dim=1)
                    batch_one_hot_label = torch.zeros(
                        len(self.y_val[(i) * int(len(self.x_val) / 10): (i + 1) * int(len(self.x_val) / 10)]),
                        self.num_classes).to(self.device)
                    if (i == 0):
                        scores = batch_scores
                        one_hot_label = batch_one_hot_label.scatter_(1, self.y_val[
                                                                        (i) * int(len(self.x_val) / 10): (i + 1) * int(
                                                                            len(self.x_val) / 10)].view(-1, 1), 1)
                    else:
                        scores = torch.cat((scores, batch_scores), dim=0)
                        one_hot_label = torch.cat((one_hot_label, batch_one_hot_label.scatter_(1, self.y_val[(i) * int(
                            len(self.x_val) / 10): (i + 1) * int(len(self.x_val) / 10)].view(-1, 1), 1)), dim=0)
                grads = scores - one_hot_label
                grads[0:int(self.facloc_size)] = self.lam * grads[0:int(self.facloc_size)]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(self.eta * grads_currX)
                for i in range(10):
                    batch_scores = F.softmax(self.model(
                        self.x_val[(i) * int((len(self.x_val)) / 10): (i + 1) * int((len(self.x_val)) / 10)]), dim=1)
                    batch_one_hot_label = torch.zeros(
                        len(self.y_val[(i) * int(len(self.x_val) / 10): (i + 1) * int(len(self.x_val) / 10)]),
                        self.num_classes).to(self.device)
                    if (i == 0):
                        scores = batch_scores
                        one_hot_label = batch_one_hot_label.scatter_(1, self.y_val[
                                                                        (i) * int(len(self.x_val) / 10): (i + 1) * int(
                                                                            len(self.x_val) / 10)].view(-1, 1), 1)
                    else:
                        scores = torch.cat((scores, batch_scores), dim=0)
                        one_hot_label = torch.cat((one_hot_label, batch_one_hot_label.scatter_(1, self.y_val[(i) * int(
                            len(self.x_val) / 10): (i + 1) * int(len(self.x_val) / 10)].view(-1, 1), 1)), dim=0)
                grads = scores - one_hot_label
                grads[0:int(self.facloc_size)] = self.lam * grads[0:int(self.facloc_size)]
        self.grads_val_curr = grads.mean(dim=0)  # reset parm.grads to zero!

    def eval_taylor(self, grads_elem, theta_init):
        grads_val = self.grads_val_curr
        dot_prod = 0
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            params = [param for param in self.model.parameters()]
            dot_prod += torch.sum(grads_val[0] * (params[-1].data - self.eta * grads_elem[0]))
        return dot_prod.data

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            grads_tensor = torch.cat(grads, dim=0)
            param_update = self.eta * grads_tensor
            gains = torch.matmul(param_update, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X += grads_e

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, remainList, theta_init):

        self.remainList = remainList
        start_time = time.time()
        self._compute_per_element_grads(theta_init)
        end_time = time.time()
        print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = [self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0]) for x in subset_selected]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains)]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if numSelected % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem)
            numSelected += 1
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


class PriorWeightedSetFunctionLoader(object):
    def __init__(self, trainset, x_val, y_val, facloc_size, lam, model, loss_criterion,
                 loss_nored, eta, device, num_classes, batch_size):
        self.trainset = trainset  # assume its a sequential loader.
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.facloc_size = facloc_size
        self.lam = lam
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = len(trainset)
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.batch_size = batch_size

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(np.arange(self.N_trn)), self.batch_size, drop_last=False))][0])
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [self.trainset[x][0].view(-1, 3, self.trainset[x][0].shape[1], self.trainset[x][0].shape[2]) for x in
                 batch_idx], dim=0).type(torch.float)
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if cnt == 0:
                with torch.no_grad():
                    data = F.softmax(self.model(inputs), dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                cnt = cnt + 1
            else:
                cnt = cnt + 1
                with torch.no_grad():
                    data = torch.cat((data, F.softmax(self.model(inputs), dim=1)), dim=0)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = torch.cat((outputs, tmp_tensor), dim=0)
        grads_vec = data - outputs
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = grads_vec

    def _update_grads_val(self, theta_init, label, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        label_indices = torch.where(self.y_val == label)[0]
        facloc_size = torch.where(label_indices < self.facloc_size)[0].shape[0]
        if first_init:
            with torch.no_grad():
                scores = F.softmax(self.model(self.x_val[label_indices]), dim=1)
                one_hot_label = torch.zeros(len(self.y_val[label_indices]), self.num_classes).to(self.device)
                one_hot_label[:, label] = 1
                grads = scores - one_hot_label
                grads[0:int(self.facloc_size)] = self.lam * grads[0:int(self.facloc_size)]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(self.eta * grads_currX)
                scores = F.softmax(self.model(self.x_val[label_indices]), dim=1)
                one_hot_label = torch.zeros(len(self.y_val[label_indices]), self.num_classes).to(self.device)
                one_hot_label[:, label] = 1
                grads = scores - one_hot_label
                grads[0:int(facloc_size)] = self.lam * grads[0:int(facloc_size)]
        self.grads_val_curr = grads.mean(dim=0)  # reset parm.grads to zero!

    def eval_taylor(self, grads_elem, theta_init):
        grads_val = self.grads_val_curr
        dot_prod = 0
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            params = [param for param in self.model.parameters()]
            dot_prod += torch.sum(grads_val[0] * (params[-1].data - self.eta * grads_elem[0]))
        return dot_prod.data

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            grads_tensor = torch.cat(grads, dim=0)
            param_update = self.eta * grads_tensor
            gains = torch.matmul(param_update, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X += grads_e

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init):
        start_time = time.time()
        self._compute_per_element_grads(theta_init)
        end_time = time.time()
        print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        t_ng_start = time.time()  # naive greedy start time
        # subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        per_class_budget = int(budget / self.num_classes)
        totalSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        for i in range(self.num_classes):
            if totalSelected == 0:
                self._update_grads_val(theta_init, label=i, first_init=True)
            else:
                self._update_grads_val(theta_init, i, grads_currX)
            numSelected = 0
            greedySet = list()
            remainSet = [x.item() for x in
                         torch.where(torch.tensor(self.trainset.dataset.targets)[self.trainset.indices] == i)[0]]
            subset_size = int((len(remainSet) / per_class_budget) * math.log(100))
            while (numSelected < per_class_budget):
                # Try Using a List comprehension here!
                t_one_elem = time.time()
                subset_selected = remainSet  # list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
                rem_grads = [self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0]) for x in subset_selected]
                gains = self.eval_taylor_modular(rem_grads, theta_init)
                # Update the greedy set and remaining set
                bestId = subset_selected[torch.argmax(gains)]
                greedySet.append(bestId)
                remainSet.remove(bestId)
                # Update info in grads_currX using element=bestId
                if totalSelected > 0:
                    self._update_gradients_subset(grads_currX, bestId)
                else:  # If 1st selection, then just set it to bestId grads
                    grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
                # Update the grads_val_current using current greedySet grads
                self._update_grads_val(theta_init, i, grads_currX)
                if numSelected % 1000 == 0:
                    # Printing bestGain and Selection time for 1 element.
                    print("numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem)
                numSelected += 1
                totalSelected += 1
            print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX