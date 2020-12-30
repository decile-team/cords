import numpy as np
import time
import torch
import math
import torch.nn.functional as F
from torch.utils.data import SequentialSampler, BatchSampler


class GlisterSetFunction(object):
    def __init__(self, trainset, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_channels, num_classes, batch_size):
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
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.numSelected = 0

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array([list(BatchSampler(SequentialSampler(np.arange(self.N_trn)),
                                                         self.batch_size, drop_last=False))][0])
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat([self.trainset[x][0].view(-1, self.num_channels, self.trainset[x][0].shape[1],
                                                         self.trainset[x][0].shape[2]) for x in batch_idx],
                               dim=0).type(torch.float)
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
                scores = F.softmax(self.model(self.x_val), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(1 * self.eta * grads_currX)
                scores = F.softmax(self.model(self.x_val), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        self.grads_val_curr = grads.mean(dim=0).view(-1, 1)  # reset parm.grads to zero!

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        # self.model.load_state_dict(theta_init)
        with torch.no_grad():
            # grads_tensor = torch.cat(grads, dim=0)
            # param_update = self.eta * grads_tensor
            gains = torch.matmul(grads, grads_val)
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
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]
            # [self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0]) for x in subset_selected]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains).item()]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            self.numSelected += 1
            # Update info in grads_currX using element=bestId
            if self.numSelected > 1:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if (self.numSelected - 1) % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


class GlisterSetFunction_Closed(object):
    def __init__(self, trainset, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_channels, num_classes, batch_size):
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
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.init_val_scores = None
        self.numSelected = 0
        self.grads_val_curr = None

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(np.arange(self.N_trn)), self.batch_size, drop_last=False))][0])
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat([self.trainset[x][0].view(-1, self.num_channels, self.trainset[x][0].shape[1],
                                                         self.trainset[x][0].shape[2]) for x in batch_idx], dim=0).type(
                torch.float)
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
                self.init_val_scores = self.model(self.x_val)
                scores = F.softmax(self.init_val_scores, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                scores = F.softmax(self.init_val_scores - ((self.eta) * grads_currX).view(1, -1), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        self.grads_val_curr = grads.mean(dim=0).view(-1, 1)  # reset parm.grads to zero!

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            # param_update = self.eta * grads
            gains = torch.matmul(grads, grads_val)
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
        # print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        # print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]
            # [self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0]) for x in subset_selected]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains).item()]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            self.numSelected += 1
            # Update info in grads_currX using element=bestId
            if self.numSelected > 1:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX=grads_currX)
            # if (self.numSelected - 1) % 1000 == 0:
            # Printing bestGain and Selection time for 1 element.
            #    print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        # print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


class Small_GlisterSetFunction(object):
    def __init__(self, x_trn, y_trn, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_trn = x_trn.to(device)
        self.y_trn = y_trn.to(device)
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = x_trn.shape[0]
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.numSelected = 0

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            scores = F.softmax(self.model(self.x_trn), dim=1)
            one_hot_label = torch.zeros(len(self.y_trn), self.num_classes).to(self.device)
            one_hot_label.scatter_(1, self.y_trn.view(-1, 1), 1)
            grads = scores - one_hot_label
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = grads

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
                params[-1].data.sub_(1 * self.eta * grads_currX)
                scores = F.softmax(self.model(self.x_val), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        self.grads_val_curr = grads.mean(dim=0).view(-1, 1)  # reset parm.grads to zero!

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            # grads_tensor = torch.cat(grads, dim=0)
            param_update = self.eta * grads
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
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]
            # [self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0]) for x in subset_selected]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains).item()]
            # remainSet[torch.argmax(gains)]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            self.numSelected += 1
            # Update info in grads_currX using element=bestId
            if self.numSelected > 1:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if (self.numSelected - 1) % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)


class Small_GlisterSetFunction_Closed(object):
    def __init__(self, x_trn, y_trn, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_trn = x_trn.to(device)
        self.y_trn = y_trn.to(device)
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = x_trn.shape[0]
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.init_val_scores = None
        self.numSelected = 0
        self.grads_val_curr = None

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            scores = F.softmax(self.model(self.x_trn), dim=1)
            one_hot_label = torch.zeros(len(self.y_trn), self.num_classes).to(self.device)
            one_hot_label.scatter_(1, self.y_trn.view(-1, 1), 1)
            grads = scores - one_hot_label
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = grads

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            with torch.no_grad():
                self.init_val_scores = self.model(self.x_val)
                scores = F.softmax(self.init_val_scores, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                scores = F.softmax(self.init_val_scores - ((self.eta) * grads_currX).view(1, -1), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        self.grads_val_curr = grads.mean(dim=0).view(-1, 1)  # reset parm.grads to zero!

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            gains = torch.matmul(grads, grads_val)
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
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains).item()]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            self.numSelected += 1
            # Update info in grads_currX using element=bestId
            if self.numSelected > 1:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX=grads_currX)
            if (self.numSelected - 1) % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)


class NonDeepSetRmodularFunction(object):
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
            inputs = torch.cat([self.trainset[x][0].view(1, -1) for x in batch_idx], dim=0).type(torch.float)
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
        #print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = grads_vec

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            with torch.no_grad():
                self.init_val_scores = self.model(self.x_val)
                scores = F.softmax(self.init_val_scores, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                scores = F.softmax(self.init_val_scores - ((self.eta) * grads_currX).view(1, -1), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        self.grads_val_curr = grads.mean(dim=0).view(-1, 1)  # reset parm.grads to zero!

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
            param_update = self.eta * grads
            gains = torch.matmul(param_update, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, indices):
        for idx in indices:
            grads_e = self.grads_per_elem[idx]
            grads_X += grads_e

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, r, theta_init):
        start_time = time.time()
        self._compute_per_element_grads(theta_init)
        end_time = time.time()
        #print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        #print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        # subset_size = int(budget/r)
        subset_size = int((len(self.grads_per_elem) / r))
        selection_size = int(budget/r)
        while (numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            sorted_gains, indices = torch.sort(gains.view(-1), descending=True)
            selected_indices = [subset_selected[index.item()] for index in indices[0:selection_size]]
            greedySet.extend(selected_indices)
            [remainSet.remove(idx) for idx in selected_indices]
            #numSelected += selection_size
            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, selected_indices)
            else:  # If 1st selection, then just set it to bestId grads
                for i in range(len(selected_indices)):
                    if i == 0:
                        grads_currX = self.grads_per_elem[selected_indices[i]]
                    else:
                        grads_e = self.grads_per_elem[selected_indices[i]]
                        grads_currX += grads_e
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if numSelected % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem)
            numSelected += selection_size
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)


class Glister_Linear_SetFunction(object):
    def __init__(self, trainset, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_channels, num_classes, batch_size):
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
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.numSelected = 0

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        embDim = self.model.get_embedding_dim()
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(np.arange(self.N_trn)),
                               self.batch_size, drop_last=False))][0])
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat([self.trainset[x][0].view(-1, self.num_channels, self.trainset[x][0].shape[1],
                                                         self.trainset[x][0].shape[2]) for x in batch_idx], dim=0).type(
                torch.float)
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if cnt == 0:
                with torch.no_grad():
                    out, l1 = self.model(inputs, last=True)
                    data = F.softmax(out, dim=1)
                l1_grads = torch.zeros(self.batch_size, embDim * self.num_classes).to(self.device)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                l0_grads = data - outputs
                # l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.batch_size):
                    for j in range(self.num_classes):
                        l1_grads[i][j * (embDim): (j + 1) * embDim] = l0_grads[i, j] * l1[i]
                cnt = cnt + 1
            else:
                with torch.no_grad():
                    out, l1 = self.model(inputs, last=True)
                    data = F.softmax(out, dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                batch_l0_grads = data - outputs
                batch_l1_grads = torch.zeros(self.batch_size, embDim * self.num_classes).to(self.device)
                # batch_l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.batch_size):
                    for j in range(self.num_classes):
                        batch_l1_grads[i][j * (embDim): (j + 1) * embDim] = batch_l0_grads[i, j] * l1[i]
                        # batch_l1_grads[i, j, :] = batch_l0_grads[i, j] * l1[i]
                l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
                cnt = cnt + 1
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        # grads_list = list()
        # for i in range(l0_grads.shape[0]):
        #    grads_list.append([l0_grads[i], l1_grads[i]])
        self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            with torch.no_grad():
                out, l1 = self.model(self.x_val, last=True)
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                embDim = self.model.get_embedding_dim()
                l1_grads = torch.zeros(l1.shape[0], self.num_classes * embDim).to(self.device)
                # l1_grads = torch.zeros(l1.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(l1.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i][(j * embDim):((j + 1) * embDim)] = l0_grads[i, j] * l1[i]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(1 * self.eta * grads_currX[0][0:self.num_classes])
                for j in range(self.num_classes):
                    params[-2].data[j].sub_(1 * self.eta * grads_currX[0][(j * embDim) + self.num_classes:((
                                                                                                                       j + 1) * embDim) + self.num_classes])
                out, l1 = self.model(self.x_val, last=True)
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l1_grads = torch.zeros(l1.shape[0], self.num_classes * embDim).to(self.device)
                # l1_grads = torch.zeros(l1.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(l1.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i][(j * embDim):((j + 1) * embDim)] = l0_grads[i, j] * l1[i]
        self.grads_val_curr = torch.mean(torch.cat((l0_grads, l1_grads), dim=1), dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            # grads_tensor = torch.cat(grads, dim=0)
            # param_update = self.eta * grads
            gains = torch.matmul(grads, grads_val)
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
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]
            gains = self.eval_taylor_modular(rem_grads)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains).item()]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            self.numSelected += 1
            # Update info in grads_currX using element=bestId
            if self.numSelected > 1:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId].view(1, -1)  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if (self.numSelected - 1) % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


class Glister_Linear_SetFunction_Closed(object):
    def __init__(self, trainset, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_channels, num_classes, batch_size):
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
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.numSelected = 0

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        embDim = self.model.get_embedding_dim()
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(np.arange(self.N_trn)),
                               self.batch_size, drop_last=False))][0])
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat([self.trainset[x][0].view(-1, self.num_channels, self.trainset[x][0].shape[1],
                                                         self.trainset[x][0].shape[2]) for x in batch_idx], dim=0).type(
                torch.float)
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if cnt == 0:
                with torch.no_grad():
                    out, l1 = self.model(inputs, last=True)
                    data = F.softmax(out, dim=1)
                l1_grads = torch.zeros(self.batch_size, embDim * self.num_classes).to(self.device)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                l0_grads = data - outputs
                # l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.batch_size):
                    for j in range(self.num_classes):
                        l1_grads[i][j * (embDim): (j + 1) * embDim] = l0_grads[i, j] * l1[i]
                cnt = cnt + 1
            else:
                with torch.no_grad():
                    out, l1 = self.model(inputs, last=True)
                    data = F.softmax(out, dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                batch_l0_grads = data - outputs
                batch_l1_grads = torch.zeros(self.batch_size, embDim * self.num_classes).to(self.device)
                # batch_l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.batch_size):
                    for j in range(self.num_classes):
                        batch_l1_grads[i][j * (embDim): (j + 1) * embDim] = batch_l0_grads[i, j] * l1[i]
                        # batch_l1_grads[i, j, :] = batch_l0_grads[i, j] * l1[i]
                l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
                cnt = cnt + 1
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        # grads_list = list()
        # for i in range(l0_grads.shape[0]):
        #    grads_list.append([l0_grads[i], l1_grads[i]])
        self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        if first_init:
            with torch.no_grad():
                self.init_out, self.init_l1 = self.model(self.x_val, last=True)
                scores = F.softmax(self.init_out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                embDim = self.model.get_embedding_dim()
                l1_grads = torch.zeros(self.init_l1.shape[0], self.num_classes * embDim).to(self.device)
                # l1_grads = torch.zeros(l1.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.init_l1.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i][(j * embDim):((j + 1) * embDim)] = l0_grads[i, j] * self.init_l1[i]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(1 * self.eta * grads_currX[0][0:self.num_classes])
                out = torch.zeros(self.init_out.shape[0], self.init_out.shape[1]).to(self.device)
                for j in range(self.num_classes):
                    out[:, j] = self.init_out[:, j] - (1 * self.eta * (torch.matmul(self.init_l1, grads_currX[0][(
                                                                                                                             j * embDim) + self.num_classes:(
                                                                                                                                                                        (
                                                                                                                                                                                    j + 1) * embDim) + self.num_classes].view(
                        -1, 1)) + grads_currX[0][j])).view(-1)
                    # params[-2].data[j].sub_(1 * self.eta * grads_currX[0][(j*embDim)+10:((j+1)*embDim)+10])
                # out, l1 = self.model(self.x_val, last=True)
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l1_grads = torch.zeros(self.init_l1.shape[0], self.num_classes * embDim).to(self.device)
                # l1_grads = torch.zeros(l1.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.init_l1.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i][(j * embDim):((j + 1) * embDim)] = l0_grads[i, j] * self.init_l1[i]
        self.grads_val_curr = torch.mean(torch.cat((l0_grads, l1_grads), dim=1), dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            # grads_tensor = torch.cat(grads, dim=0)
            # param_update = self.eta * grads
            gains = torch.matmul(grads, grads_val)
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
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            # subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[remainSet]
            gains = self.eval_taylor_modular(rem_grads)
            # Update the greedy set and remaining set
            bestId = remainSet[torch.argmax(gains).item()]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            self.numSelected += 1
            # Update info in grads_currX using element=bestId
            if self.numSelected > 1:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId].view(1, -1)  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if (self.numSelected - 1) % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


class Small_Glister_Linear_SetFunction(object):
    def __init__(self, x_trn, y_trn, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_trn = x_trn.to(device)
        self.y_trn = y_trn.to(device)
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = x_trn.shape[0]
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.numSelected = 0

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        embDim = self.model.get_embedding_dim()
        with torch.no_grad():
            out, l1 = self.model(self.x_trn, last=True)
            data = F.softmax(out, dim=1)
        l1_grads = torch.zeros(self.N_trn, embDim * self.num_classes).to(self.device)
        tmp_tensor = torch.zeros(self.N_trn, self.num_classes).to(self.device)
        tmp_tensor.scatter_(1, self.y_trn.view(-1, 1), 1)
        outputs = tmp_tensor
        l0_grads = data - outputs
        # l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
        for i in range(self.N_trn):
            for j in range(self.num_classes):
                l1_grads[i][(j * embDim): ((j + 1) * embDim)] = l0_grads[i, j] * l1[i]
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            with torch.no_grad():
                out, l1 = self.model(self.x_val, last=True)
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                embDim = self.model.get_embedding_dim()
                l1_grads = torch.zeros(l1.shape[0], self.num_classes * embDim).to(self.device)
                # l1_grads = torch.zeros(l1.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(l1.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i][(j * embDim):((j + 1) * embDim)] = l0_grads[i, j] * l1[i]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(1 * self.eta * grads_currX[0][0:self.num_classes])
                for j in range(self.num_classes):
                    params[-2].data[j].sub_(1 * self.eta * grads_currX[0][((j * embDim) + \
                                                                           self.num_classes):(((
                                                                                                           j + 1) * embDim) + self.num_classes)])
                out, l1 = self.model(self.x_val, last=True)
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l1_grads = torch.zeros(l1.shape[0], self.num_classes * embDim).to(self.device)
                # l1_grads = torch.zeros(l1.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(l1.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i][(j * embDim):((j + 1) * embDim)] = l0_grads[i, j] * l1[i]
        self.grads_val_curr = torch.mean(torch.cat((l0_grads, l1_grads), dim=1), dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            # grads_tensor = torch.cat(grads, dim=0)
            # param_update = self.eta * grads
            gains = torch.matmul(grads, grads_val)
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
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]
            gains = self.eval_taylor_modular(rem_grads)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains).item()]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            self.numSelected += 1
            # Update info in grads_currX using element=bestId
            if self.numSelected > 1:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId].view(1, -1)  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if (self.numSelected - 1) % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)


class Small_Glister_Linear_SetFunction_Closed(object):
    def __init__(self, x_trn, y_trn, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_trn = x_trn.to(device)
        self.y_trn = y_trn.to(device)
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = x_trn.shape[0]
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.numSelected = 0

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        embDim = self.model.get_embedding_dim()
        with torch.no_grad():
            out, l1 = self.model(self.x_trn, last=True)
            data = F.softmax(out, dim=1)
        l1_grads = torch.zeros(self.N_trn, embDim * self.num_classes).to(self.device)
        tmp_tensor = torch.zeros(self.N_trn, self.num_classes).to(self.device)
        tmp_tensor.scatter_(1, self.y_trn.view(-1, 1), 1)
        outputs = tmp_tensor
        l0_grads = data - outputs
        # l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
        for i in range(self.N_trn):
            for j in range(self.num_classes):
                l1_grads[i][(j * embDim): ((j + 1) * embDim)] = l0_grads[i, j] * l1[i]
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            with torch.no_grad():
                self.init_out, self.init_l1 = self.model(self.x_val, last=True)
                scores = F.softmax(self.init_out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                embDim = self.model.get_embedding_dim()
                l1_grads = torch.zeros(self.init_l1.shape[0], self.num_classes * embDim).to(self.device)
                # l1_grads = torch.zeros(l1.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.init_l1.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i, (j * embDim):((j + 1) * embDim)] = l0_grads[i, j] * self.init_l1[i]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                out = torch.zeros(self.init_out.shape[0], self.num_classes).to(self.device)
                for j in range(self.num_classes):
                    out[:, j] = self.init_out[:, j] - (1 * self.eta * (torch.matmul(self.init_l1, grads_currX[0][(
                                                                                                                             j * embDim) + self.num_classes:(
                                                                                                                                                                        (
                                                                                                                                                                                    j + 1) * embDim) + self.num_classes].view(
                        -1, 1)) + grads_currX[0][j])).view(-1)
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l1_grads = torch.zeros(self.init_l1.shape[0], self.num_classes * embDim).to(self.device)
                # l1_grads = torch.zeros(l1.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.init_l1.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i][(j * embDim):((j + 1) * embDim)] = l0_grads[i, j] * self.init_l1[i]
        self.grads_val_curr = torch.cat((l0_grads, l1_grads), dim=1).mean(dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            # grads_tensor = torch.cat(grads, dim=0)
            # param_update = self.eta * grads
            gains = torch.matmul(grads, grads_val)
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
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]
            gains = self.eval_taylor_modular(rem_grads)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains).item()]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            self.numSelected += 1
            # Update info in grads_currX using element=bestId
            if self.numSelected > 1:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId].view(1, -1)  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if (self.numSelected - 1) % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)


class Small_Glister_Linear_SetFunction_RModular(object):
    def __init__(self, x_trn, y_trn, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_trn = x_trn.to(device)
        self.y_trn = y_trn.to(device)
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = x_trn.shape[0]
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.numSelected = 0

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        embDim = self.model.get_embedding_dim()
        with torch.no_grad():
            out, l1 = self.model(self.x_trn, last=True)
            data = F.softmax(out, dim=1)
        tmp_tensor = torch.zeros(self.x_trn.shape[0], self.num_classes).to(self.device)
        tmp_tensor.scatter_(1, self.y_trn.view(-1, 1), 1)
        outputs = tmp_tensor
        l0_grads = data - outputs
        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
        torch.cuda.empty_cache()
        self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        if first_init:
            with torch.no_grad():
                self.init_out, self.init_l1 = self.model(self.x_val, last=True)
                scores = F.softmax(self.init_out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                embDim = self.model.get_embedding_dim()
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
                torch.cuda.empty_cache()


        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                out = torch.zeros(self.init_out.shape[0], self.num_classes).to(self.device)
                for j in range(self.num_classes):
                    out[:, j] = self.init_out[:, j] - (1 * self.eta * (torch.matmul(self.init_l1, grads_currX[0][(j * embDim) + self.num_classes:((j + 1) * embDim) + self.num_classes].view(-1, 1)) + grads_currX[0][j])).view(-1)
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
                torch.cuda.empty_cache()
        self.grads_val_curr = torch.cat((l0_grads, l1_grads), dim=1).mean(dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            gains = torch.matmul(grads, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, indices):
        for idx in indices:
            grads_e = self.grads_per_elem[idx]
            grads_X += grads_e

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init, r=10):
        #start_time = time.time()
        self._compute_per_element_grads(theta_init)
        #end_time = time.time()
        #print("Per Element gradient computation time is: ", end_time - start_time)
        #start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        #end_time = time.time()
        #print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int(budget / r)
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            #subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[remainSet]
            gains = self.eval_taylor_modular(rem_grads)
            # Update the greedy set and remaining set
            # Update the greedy set and remaining set
            sorted_gains, indices = torch.sort(gains.view(-1), descending=True)
            selected_indices = [remainSet[index.item()] for index in indices[0:subset_size]]
            greedySet.extend(selected_indices)
            [remainSet.remove(idx) for idx in selected_indices]
            self.numSelected += subset_size
            # Update info in grads_currX using element=bestId
            if self.numSelected > subset_size:
                self._update_gradients_subset(grads_currX, selected_indices)
            else:  # If 1st selection, then just set it to bestId grads
                for i in range(len(selected_indices)):
                    if i == 0:
                        grads_currX = self.grads_per_elem[selected_indices[i]].view(1, -1)
                    else:
                        grads_e = self.grads_per_elem[selected_indices[i]]
                        grads_currX += grads_e
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if (self.numSelected - 1) % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)


class Small_GLISTER_WeightedSetFunction(object):
    def __init__(self, x_trn, y_trn, x_val, y_val, facloc_size, lam, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_trn = x_trn.to(device)
        self.y_trn = y_trn.to(device)
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
        self.N_trn = x_trn.shape[0]
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.numSelected = 0

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            scores = F.softmax(self.model(self.x_trn), dim=1)
            one_hot_label = torch.zeros(len(self.y_trn), self.num_classes).to(self.device)
            one_hot_label.scatter_(1, self.y_trn.view(-1, 1), 1)
            grads = scores - one_hot_label
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = grads

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
                params[-1].data.sub_(1 * self.eta * grads_currX)
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
        self.grads_val_curr = grads.mean(dim=0).view(-1, 1)  # reset parm.grads to zero!

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            gains = torch.matmul(grads, grads_val)
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
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains).item()]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            self.numSelected += 1
            # Update info in grads_currX using element=bestId
            if self.numSelected > 1:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if self.numSelected % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)


class GLISTER_WeightedSetFunction(object):
    def __init__(self, trainset, x_val, y_val, facloc_size, lam, model, loss_criterion,
                 loss_nored, eta, device, num_channels, num_classes, batch_size):
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
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.numSelected = 0

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(np.arange(self.N_trn)), self.batch_size, drop_last=False))][0])
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [self.trainset[x][0].view(-1, self.num_channels, self.trainset[x][0].shape[1],
                                          self.trainset[x][0].shape[2]) for x in
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
                params[-1].data.sub_(1 * self.eta * grads_currX)
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
    def naive_greedy_max(self, budget, theta_init):
        start_time = time.time()
        self._compute_per_element_grads(theta_init)
        end_time = time.time()
        print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = [self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0]) for x in subset_selected]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains)]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            self.numSelected += 1
            # Update info in grads_currX using element=bestId
            if self.numSelected > 1:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if self.numSelected % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


class GLISTER_Linear_WeightedSetFunction(object):
    def __init__(self, trainset, x_val, y_val, facloc_size, lam, model, loss_criterion,
                 loss_nored, eta, device, num_channels, num_classes, batch_size):
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
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(np.arange(self.N_trn)), self.batch_size, drop_last=False))][0])
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [self.trainset[x][0].view(-1, self.num_channels, self.trainset[x][0].shape[1],
                                          self.trainset[x][0].shape[2])
                 for x in batch_idx], dim=0).type(torch.float)
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if cnt == 0:
                with torch.no_grad():
                    out, l1 = self.model(inputs)
                    data = F.softmax(out, dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                l0_grads = data - outputs
                l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.batch_size):
                    for j in range(self.num_classes):
                        l1_grads[i, j, :] = l0_grads[i, j] * l1[i]
                cnt = cnt + 1
            else:
                with torch.no_grad():
                    out, l1 = self.model(inputs)
                    data = F.softmax(out, dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                batch_l0_grads = data - outputs
                batch_l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.batch_size):
                    for j in range(self.num_classes):
                        batch_l1_grads[i, j, :] = batch_l0_grads[i, j] * l1[i]
                l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
                cnt = cnt + 1
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        grads_list = list()
        for i in range(l0_grads.shape[0]):
            grads_list.append([l0_grads[i], l1_grads[i]])
        self.grads_per_elem = grads_list

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            with torch.no_grad():
                for i in range(10):
                    batch_out, batch_l1 = self.model(
                        self.x_val[(i) * int((len(self.x_val)) / 10): (i + 1) * int((len(self.x_val)) / 10)])
                    batch_scores = F.softmax(batch_out, dim=1)
                    batch_one_hot_label = torch.zeros(
                        len(self.y_val[(i) * int(len(self.x_val) / 10): (i + 1) * int(len(self.x_val) / 10)]),
                        self.num_classes).to(self.device)
                    if (i == 0):
                        scores = batch_scores
                        l1 = batch_l1
                        one_hot_label = batch_one_hot_label.scatter_(1, self.y_val[
                                                                        (i) * int(len(self.x_val) / 10): (i + 1) * int(
                                                                            len(self.x_val) / 10)].view(-1, 1), 1)
                    else:
                        scores = torch.cat((scores, batch_scores), dim=0)
                        l1 = torch.cat((scores, batch_l1), dim=0)
                        one_hot_label = torch.cat((one_hot_label, batch_one_hot_label.scatter_(1, self.y_val[(i) * int(
                            len(self.x_val) / 10): (i + 1) * int(len(self.x_val) / 10)].view(-1, 1), 1)), dim=0)
                l0_grads = scores - one_hot_label
                l0_grads[0:int(self.facloc_size)] = self.lam * l0_grads[0:int(self.facloc_size)]
                l1_grads = torch.zeros(l0_grads.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(l0_grads.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i, j, :] = l0_grads[i, j] * l1[i]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(self.eta * grads_currX[0])
                params[-2].data.sub_(self.eta * grads_currX[1])
                for i in range(10):
                    batch_out, batch_l1 = self.model(
                        self.x_val[(i) * int((len(self.x_val)) / 10): (i + 1) * int((len(self.x_val)) / 10)])
                    batch_scores = F.softmax(batch_out, dim=1)
                    batch_one_hot_label = torch.zeros(
                        len(self.y_val[(i) * int(len(self.x_val) / 10): (i + 1) * int(len(self.x_val) / 10)]),
                        self.num_classes).to(self.device)
                    if (i == 0):
                        scores = batch_scores
                        l1 = batch_l1
                        one_hot_label = batch_one_hot_label.scatter_(1, self.y_val[
                                                                        (i) * int(len(self.x_val) / 10): (i + 1) * int(
                                                                            len(self.x_val) / 10)].view(-1, 1), 1)
                    else:
                        scores = torch.cat((scores, batch_scores), dim=0)
                        l1 = torch.cat((scores, batch_l1), dim=0)
                        one_hot_label = torch.cat((one_hot_label, batch_one_hot_label.scatter_(1, self.y_val[(i) * int(
                            len(self.x_val) / 10): (i + 1) * int(len(self.x_val) / 10)].view(-1, 1), 1)), dim=0)
                l0_grads = scores - one_hot_label
                l0_grads[0:int(self.facloc_size)] = self.lam * l0_grads[0:int(self.facloc_size)]
                l1_grads = torch.zeros(l0_grads.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(l0_grads.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i, j, :] = l0_grads[i, j] * l1[i]
        self.grads_val_curr = torch.cat((torch.flatten(l0_grads.mean(dim=0)), torch.flatten(l1_grads.mean(dim=0))),
                                        dim=0)

    def eval_taylor_modular(self, grads):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            grads_tensor = torch.cat(grads, dim=0)
            param_update = self.eta * grads_tensor
            gains = torch.matmul(param_update, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X[0] += grads_e[0]
        grads_X[1] += grads_e[1]

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init):
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
            rem_grads = [torch.cat((torch.flatten(self.grads_per_elem[x][0]), torch.flatten(self.grads_per_elem[x][1])),
                                   dim=0).reshape(1, -1) for x in subset_selected]
            gains = self.eval_taylor_modular(rem_grads)
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
