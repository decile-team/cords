import math
import numpy as np
import time
import torch
import torch.nn.functional as F
from queue import PriorityQueue

class Small_GlisterAct_SetFunction(object):
    def __init__(self, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        #self.N_trn = x_trn.shape[0]
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.numSelected = 0

    def _compute_per_element_grads(self, x_trn, y_trn, theta_init):
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            scores = F.softmax(self.model(x_trn), dim=1)
            one_hot_label = torch.zeros(len(y_trn), self.num_classes).to(self.device)
            one_hot_label.scatter_(1, y_trn.view(-1, 1), 1)
            grads = scores - one_hot_label
        torch.cuda.empty_cache()
        #print("Per Element Gradient Computation is Completed")
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

    def eval_taylor_modular(self, grads):
        grads_val = self.grads_val_curr
        with torch.no_grad():
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
    def naive_greedy_max(self, x_trn, y_trn, budget, theta_init):
        x_trn = x_trn.to(self.device)
        y_trn = y_trn.to(self.device)
        start_time = time.time()
        self._compute_per_element_grads(x_trn, y_trn, theta_init)
        end_time = time.time()
        #print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        #print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(x_trn.shape[0]))
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
            #remainSet[torch.argmax(gains)]
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
        #    if (self.numSelected - 1) % 1000 == 0:
        #        # Printing bestGain and Selection time for 1 element.
        #        print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        #print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)


class Small_GlisterAct_SetFunction_Closed(object):
    def __init__(self, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.init_val_scores = None
        self.numSelected = 0
        self.grads_val_curr = None

    def _compute_per_element_grads(self, x_trn, y_trn, theta_init):
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            scores = F.softmax(self.model(x_trn), dim=1)
            one_hot_label = torch.zeros(len(y_trn), self.num_classes).to(self.device)
            one_hot_label.scatter_(1, y_trn.view(-1, 1), 1)
            grads = scores - one_hot_label
        torch.cuda.empty_cache()
        #print("Per Element Gradient Computation is Completed")
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
                scores = F.softmax(self.init_val_scores - ((self.eta/self.numSelected) * grads_currX).view(1, -1), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        self.grads_val_curr = grads.mean(dim=0).view(-1, 1)  # reset parm.grads to zero!

    def eval_taylor_modular(self, grads):
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
    def naive_greedy_max(self, x_trn, y_trn, budget, theta_init):
        x_trn = x_trn.to(self.device)
        y_trn = y_trn.to(self.device)
        start_time = time.time()
        self._compute_per_element_grads(x_trn, y_trn, theta_init)
        end_time = time.time()
        #print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        #print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(x_trn.shape[0]))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]
                #[self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0]) for x in subset_selected]
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
                grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX=grads_currX)
        #    if (self.numSelected - 1) % 1000 == 0:
        #        # Printing bestGain and Selection time for 1 element.
        #        print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        #print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)


class Small_GlisterAct_Linear_SetFunction(object):
    def __init__(self, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.numSelected = 0

    def _compute_per_element_grads(self, x_trn, y_trn, theta_init):
        self.model.load_state_dict(theta_init)
        embDim = self.model.get_embedding_dim()
        with torch.no_grad():
            out, l1 = self.model(x_trn, last=True)
            data = F.softmax(out, dim=1)
        l1_grads = torch.zeros(x_trn.shape[0], embDim * self.num_classes).to(self.device)
        tmp_tensor = torch.zeros(x_trn.shape[0], self.num_classes).to(self.device)
        tmp_tensor.scatter_(1, y_trn.view(-1, 1), 1)
        outputs = tmp_tensor
        l0_grads = data - outputs
        # l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
        for i in range(x_trn.shape[0]):
            for j in range(self.num_classes):
                l1_grads[i][(j * embDim): ((j + 1) * embDim)] = l0_grads[i, j] * l1[i]
        torch.cuda.empty_cache()
        #print("Per Element Gradient Computation is Completed")
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
                #l1_grads = torch.zeros(l1.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(l1.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i][(j * embDim):((j+1) * embDim)] = l0_grads[i, j] * l1[i]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(1 * self.eta * grads_currX[0][0:self.num_classes])
                for j in range(self.num_classes):
                    params[-2].data[j].sub_(1 * self.eta * grads_currX[0][((j*embDim) + \
                    self.num_classes):(((j+1)*embDim)+self.num_classes)])
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
            gains = torch.matmul(grads, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X += grads_e

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, x_trn, y_trn, budget, theta_init):
        x_trn = x_trn.to(self.device)
        y_trn = y_trn.to(self.device)
        start_time = time.time()
        self._compute_per_element_grads(x_trn, y_trn, theta_init)
        end_time = time.time()
        #print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        #print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(x_trn.shape[0]))
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
            #if (self.numSelected - 1) % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
            #   print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)

class Small_GlisterAct_Linear_SetFunction_Closed_Vect(object):
    def __init__(self, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.numSelected = 0

    def _compute_per_element_grads(self, x_trn, y_trn, x_lbl, y_lbl, theta_init):
        self.model.load_state_dict(theta_init)
        embDim = self.model.get_embedding_dim()
        with torch.no_grad():
            out, l1 = self.model(x_trn, last=True)
            data = F.softmax(out, dim=1)
        tmp_tensor = torch.zeros(x_trn.shape[0], self.num_classes).to(self.device)
        tmp_tensor.scatter_(1, y_trn.view(-1, 1), 1)
        outputs = tmp_tensor
        l0_grads = data - outputs
        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
        torch.cuda.empty_cache()
        self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)

        with torch.no_grad():
            out, l1 = self.model(x_lbl, last=True)
            data = F.softmax(out, dim=1)
        tmp_tensor = torch.zeros(x_lbl.shape[0], self.num_classes).to(self.device)
        tmp_tensor.scatter_(1, y_lbl.view(-1, 1), 1)
        outputs = tmp_tensor
        l0_grads = data - outputs
        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
        l0_grads = l0_grads.sum(dim=0).view(1, -1)
        l1_grads = l1_grads.sum(dim=0).view(1, -1)
        torch.cuda.empty_cache()
        self.prev_grads_sum = torch.cat((l0_grads, l1_grads), dim=1)

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            # with torch.no_grad():
            #     self.init_out, self.init_l1 = self.model(self.x_val, last=True)
            #     scores = F.softmax(self.init_out, dim=1)
            #     one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
            #     one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
            #     l0_grads = scores - one_hot_label
            #     embDim = self.model.get_embedding_dim()
            #     l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            #     l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                self.init_out, self.init_l1 = self.model(self.x_val, last=True)
                out = torch.zeros(self.init_out.shape[0], self.num_classes).to(self.device)
                for j in range(self.num_classes):
                    out[:, j] = self.init_out[:, j] - (1 * self.eta * (torch.matmul(self.init_l1, self.prev_grads_sum[0][(j * embDim) +
                                self.num_classes:((j + 1) * embDim) + self.num_classes].view(-1, 1)) + self.prev_grads_sum[0][j])).view(-1)
                self.init_out = out
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
        # populate the gradients in model params based on loss.

        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                out = torch.zeros(self.init_out.shape[0], self.num_classes).to(self.device)
                for j in range(self.num_classes):
                    out[:, j] = self.init_out[:, j] - (1 * self.eta * (torch.matmul(self.init_l1, grads_currX[0][(j * embDim) +
                                self.num_classes:((j + 1) * embDim) + self.num_classes].view(-1, 1)) + grads_currX[0][j])).view(-1)

                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
        self.grads_val_curr = torch.cat((l0_grads, l1_grads), dim=1).mean(dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads):
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
    def naive_greedy_max(self, x_trn, y_trn, x_lbl, y_lbl, budget, theta_init):
        start_time = time.time()
        x_trn = x_trn.to(self.device)
        y_trn = y_trn.to(self.device)
        self._compute_per_element_grads(x_trn, y_trn, x_lbl, y_lbl, theta_init)
        end_time = time.time()
        #print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        #print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(x_trn.shape[0]))
        t_ng_start = time.time()  # naive greedy start time
        #subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            '''subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]'''
            gains = self.eval_taylor_modular(self.grads_per_elem[remainSet])#rem_grads)
            # Update the greedy set and remaining set
            #bestId = subset_selected[torch.argmax(gains).item()]
            bestId = remainSet[torch.argmax(gains).item()]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            self.numSelected += 1
            # Update info in grads_currX using element=bestId
            if self.numSelected > 1:
                #self._update_gradients_subset(grads_currX, bestId)
                grads_currX += self.grads_per_elem[bestId]
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId].view(1, -1)  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            #if (self.numSelected - 1)%1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                #print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        #print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)


class Small_GlisterAct_Linear_SetFunction_Closed(object):
    def __init__(self, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.numSelected = 0

    def _compute_per_element_grads(self, x_trn, y_trn, theta_init):
        self.model.load_state_dict(theta_init)
        embDim = self.model.get_embedding_dim()
        with torch.no_grad():
            out, l1 = self.model(x_trn, last=True)
            data = F.softmax(out, dim=1)
        l1_grads = torch.zeros(x_trn.shape[0], embDim * self.num_classes).to(self.device)
        tmp_tensor = torch.zeros(x_trn.shape[0], self.num_classes).to(self.device)
        tmp_tensor.scatter_(1, y_trn.view(-1, 1), 1)
        outputs = tmp_tensor
        l0_grads = data - outputs
        # l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
        for i in range(x_trn.shape[0]):
            for j in range(self.num_classes):
                l1_grads[i][(j * embDim): ((j + 1) * embDim)] = l0_grads[i, j] * l1[i]
        torch.cuda.empty_cache()
        #print("Per Element Gradient Computation is Completed")
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
                #l1_grads = torch.zeros(l1.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.init_l1.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i, (j * embDim):((j+1) * embDim)] = l0_grads[i, j] * self.init_l1[i]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                out = torch.zeros(self.init_out.shape[0], self.num_classes).to(self.device)
                for j in range(self.num_classes):
                    out[:, j] = self.init_out[:, j] - (1 * self.eta * (torch.matmul(self.init_l1, grads_currX[0][(j*embDim)+self.num_classes:((j+1)*embDim)+self.num_classes].view(-1, 1)) + grads_currX[0][j])).view(-1)
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
            gains = torch.matmul(grads, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X += grads_e

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, x_trn, y_trn, budget, theta_init):
        start_time = time.time()
        x_trn = x_trn.to(self.device)
        y_trn = y_trn.to(self.device)
        self._compute_per_element_grads(x_trn, y_trn, theta_init)
        end_time = time.time()
        #print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        #print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(x_trn.shape[0]))
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
            #if (self.numSelected - 1)%1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                #print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        #print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)


class SDReg_GlisterActLinear_SetFunction_Closed_Vect(object):
    def __init__(self, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.numSelected = 0
        self.N = 0

    def _distance(self,x, y, exp = 2):
      n = x.size(0)
      m = y.size(0)
      d = x.size(1)
      x = x.unsqueeze(1).expand(n, m, d)
      y = y.unsqueeze(0).expand(n, m, d)
      dist = torch.exp((-1 * torch.pow(x - y, exp).sum(2))/2)
      return dist

    def _compute_similarity_kernel(self):
        train_batch_size_for_greedy = 200
        train_loader_greedy = []
        for item in range(math.ceil(len(self.grads_per_elem) / train_batch_size_for_greedy)):
            inputs = self.grads_per_elem[item * train_batch_size_for_greedy:(item + 1) * train_batch_size_for_greedy]
            train_loader_greedy.append(inputs)
        g_is = []
        with torch.no_grad():
            for i, data_i in enumerate(train_loader_greedy, 0):
                #inputs_i = inputs_i.to(self.device)  # , target_i.to(self.device)
                self.N += data_i.size()[0]
                g_is.append(data_i)

            #self.sim_mat = torch.zeros([self.N, self.N], dtype=torch.float32).to(self.device)
            new_N = len(self.grads_per_elem)
            self.sim_mat = torch.zeros([new_N, new_N], dtype=torch.float32)#.to(self.device)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.sim_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self._distance(g_i, g_j)


    def _compute_per_element_grads(self, x_trn, y_trn, x_lbl, y_lbl, theta_init):
        self.model.load_state_dict(theta_init)
        embDim = self.model.get_embedding_dim()
        with torch.no_grad():
            out, l1 = self.model(x_trn, last=True)
            data = F.softmax(out, dim=1)
        tmp_tensor = torch.zeros(x_trn.shape[0], self.num_classes).to(self.device)
        tmp_tensor.scatter_(1, y_trn.view(-1, 1), 1)
        outputs = tmp_tensor
        l0_grads = data - outputs
        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
        torch.cuda.empty_cache()
        self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)

        with torch.no_grad():
            out, l1 = self.model(x_lbl, last=True)
            data = F.softmax(out, dim=1)
        tmp_tensor = torch.zeros(x_lbl.shape[0], self.num_classes).to(self.device)
        tmp_tensor.scatter_(1, y_lbl.view(-1, 1), 1)
        outputs = tmp_tensor
        l0_grads = data - outputs
        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
        l0_grads = l0_grads.sum(dim=0).view(1, -1)
        l1_grads = l1_grads.sum(dim=0).view(1, -1)
        torch.cuda.empty_cache()
        self.prev_grads_sum = torch.cat((l0_grads, l1_grads), dim=1)

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            # with torch.no_grad():
            #     self.init_out, self.init_l1 = self.model(self.x_val, last=True)
            #     scores = F.softmax(self.init_out, dim=1)
            #     one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
            #     one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
            #     l0_grads = scores - one_hot_label
            #     embDim = self.model.get_embedding_dim()
            #     l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            #     l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                self.init_out, self.init_l1 = self.model(self.x_val, last=True)
                out = torch.zeros(self.init_out.shape[0], self.num_classes).to(self.device)
                for j in range(self.num_classes):
                    out[:, j] = self.init_out[:, j] - (1 * self.eta * (torch.matmul(self.init_l1, self.prev_grads_sum[0][(j * embDim) +
                                self.num_classes:((j + 1) * embDim) + self.num_classes].view(-1, 1)) + self.prev_grads_sum[0][j])).view(-1)
                self.init_out = out
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
        # populate the gradients in model params based on loss.

        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                out = torch.zeros(self.init_out.shape[0], self.num_classes).to(self.device)
                for j in range(self.num_classes):
                    out[:, j] = self.init_out[:, j] - (1 * self.eta * (torch.matmul(self.init_l1, grads_currX[0][(j * embDim) +
                                self.num_classes:((j + 1) * embDim) + self.num_classes].view(-1, 1)) + grads_currX[0][j])).view(-1)

                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
        self.grads_val_curr = torch.cat((l0_grads, l1_grads), dim=1).mean(dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads, greedySet,subset):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            if len(greedySet) > 0:
                #gains = torch.matmul(grads, grads_val) - 10*self.sim_mat[subset][:, greedySet].sum(1).view(-1, 1).to(self.device)
                gains = torch.matmul(grads, grads_val) - 10*self.sim_mat[subset][:, greedySet].sum(1).view(-1, 1).to(self.device)
            else:
                gains = torch.matmul(grads, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X += grads_e

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, x_trn, y_trn, x_lbl, y_lbl, budget, theta_init):
        start_time = time.time()
        x_trn = x_trn.to(self.device)
        y_trn = y_trn.to(self.device)
        self._compute_per_element_grads(x_trn, y_trn, x_lbl, y_lbl, theta_init)
        self._compute_similarity_kernel()
        end_time = time.time()
        #print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        #print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(x_trn.shape[0]))
        t_ng_start = time.time()  # naive greedy start time
        #subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            '''subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]
            gains = self.eval_taylor_modular(rem_grads, greedySet, subset_selected)'''
            gains = self.eval_taylor_modular(self.grads_per_elem[remainSet],greedySet,remainSet)#rem_grads)
            # Update the greedy set and remaining set
            #bestId = subset_selected[torch.argmax(gains).item()]
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
        return list(greedySet)


class SetFunctionTaylor(object):
    def __init__(self, X_val, Y_val, model, loss_criterion, loss_nored, eta,device,num_classes):
        self.x_val = X_val
        self.y_val = Y_val
        
        self.model = model
        self.loss = loss_criterion # For validation loss
        self.loss_nored = loss_nored # Make sure it has reduction='none' instead of default
        self.eta = eta # step size for the one step gradient update
        self.grads_per_elem = None
        self.grads_val_curr = None
        self.device = device
        self.num_classes = num_classes


    def _compute_per_element_grads(self, theta_init,x_trn,y_trn):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        scores = self.model(x_trn)
        losses = self.loss_nored(scores, y_trn)
        self.N_trn = y_trn.shape[0]
        inputs = x_trn
        targets = y_trn
        with torch.no_grad():
            data = F.softmax(self.model(inputs), dim=1)
        tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
        tmp_tensor.scatter_(1, targets.view(-1,1), 1)
        outputs = tmp_tensor
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


    def naive_greedy_max(self, x_trn, y_trn, budget, theta_init,previous=None,random=False):
        
        self.x_trn = x_trn
        self.y_trn = y_trn 

        start_time = time.time()
        self._compute_per_element_grads(theta_init,x_trn,y_trn)
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
            state = np.random.get_state()
            np.random.seed(numSelected*numSelected)
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            np.random.set_state(state)
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
        return list(greedySet)

class FacReg_GlisterActLinear_SetFunction_Closed_Vect(object):
    def __init__(self, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes):
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.numSelected = 0
        self.N = 0

    def distance(self,x, y, exp = 2):

      n = x.size(0)
      m = y.size(0)
      d = x.size(1)

      x = x.unsqueeze(1).expand(n, m, d)
      y = y.unsqueeze(0).expand(n, m, d)

      dist = torch.pow(x - y, exp).sum(2) 
      return dist 

    def _compute_similarity_kernel(self):
        train_batch_size_for_greedy = 200
        train_loader_greedy = []
        for item in range(math.ceil(len(self.grads_per_elem) / train_batch_size_for_greedy)):
            inputs = self.grads_per_elem[item * train_batch_size_for_greedy:(item + 1) * train_batch_size_for_greedy]
            train_loader_greedy.append(inputs)
        g_is = []
        with torch.no_grad():
            for i, data_i in enumerate(train_loader_greedy, 0):
                #inputs_i = inputs_i.to(self.device)  # , target_i.to(self.device)
                self.N += data_i.size()[0]
                g_is.append(data_i)

            #self.sim_mat = torch.zeros([self.N, self.N], dtype=torch.float32).to(self.device)
            new_N = len(self.grads_per_elem)
            self.sim_mat = torch.zeros([new_N, new_N], dtype=torch.float32)#.to(self.device)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.sim_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j)

            self.const = torch.max(self.sim_mat).item()
            self.sim_mat = self.const - self.sim_mat

            self.max_sim = torch.zeros(new_N, dtype=torch.float32)#.to(self.device)



    def _compute_per_element_grads(self, x_trn, y_trn, x_lbl, y_lbl, theta_init):
        self.model.load_state_dict(theta_init)
        embDim = self.model.get_embedding_dim()
        with torch.no_grad():
            out, l1 = self.model(x_trn, last=True)
            data = F.softmax(out, dim=1)
        tmp_tensor = torch.zeros(x_trn.shape[0], self.num_classes).to(self.device)
        tmp_tensor.scatter_(1, y_trn.view(-1, 1), 1)
        outputs = tmp_tensor
        l0_grads = data - outputs
        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
        torch.cuda.empty_cache()
        self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)

        with torch.no_grad():
            out, l1 = self.model(x_lbl, last=True)
            data = F.softmax(out, dim=1)
        tmp_tensor = torch.zeros(x_lbl.shape[0], self.num_classes).to(self.device)
        tmp_tensor.scatter_(1, y_lbl.view(-1, 1), 1)
        outputs = tmp_tensor
        l0_grads = data - outputs
        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
        l0_grads = l0_grads.sum(dim=0).view(1, -1)
        l1_grads = l1_grads.sum(dim=0).view(1, -1)
        torch.cuda.empty_cache()
        self.prev_grads_sum = torch.cat((l0_grads, l1_grads), dim=1)

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            # with torch.no_grad():
            #     self.init_out, self.init_l1 = self.model(self.x_val, last=True)
            #     scores = F.softmax(self.init_out, dim=1)
            #     one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
            #     one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
            #     l0_grads = scores - one_hot_label
            #     embDim = self.model.get_embedding_dim()
            #     l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            #     l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                self.init_out, self.init_l1 = self.model(self.x_val, last=True)
                out = torch.zeros(self.init_out.shape[0], self.num_classes).to(self.device)
                for j in range(self.num_classes):
                    out[:, j] = self.init_out[:, j] - (1 * self.eta * (torch.matmul(self.init_l1, self.prev_grads_sum[0][(j * embDim) +
                                self.num_classes:((j + 1) * embDim) + self.num_classes].view(-1, 1)) + self.prev_grads_sum[0][j])).view(-1)
                self.init_out = out
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
        # populate the gradients in model params based on loss.

        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                embDim = self.model.get_embedding_dim()
                out = torch.zeros(self.init_out.shape[0], self.num_classes).to(self.device)
                for j in range(self.num_classes):
                    out[:, j] = self.init_out[:, j] - (1 * self.eta * (torch.matmul(self.init_l1, grads_currX[0][(j * embDim) +
                                self.num_classes:((j + 1) * embDim) + self.num_classes].view(-1, 1)) + grads_currX[0][j])).view(-1)

                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
        self.grads_val_curr = torch.cat((l0_grads, l1_grads), dim=1).mean(dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads):#, greedySet,subset=None):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            gains = torch.matmul(grads, grads_val) +\
                 (torch.max(self.max_sim ,self.sim_mat) - self.max_sim).sum().to(self.device)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X += grads_e
        self.max_sim = torch.max(self.max_sim,self.sim_mat[element])

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, x_trn, y_trn, x_lbl, y_lbl, budget, theta_init):
        start_time = time.time()
        x_trn = x_trn.to(self.device)
        y_trn = y_trn.to(self.device)
        self._compute_per_element_grads(x_trn, y_trn, x_lbl, y_lbl, theta_init)
        self._compute_similarity_kernel()
        end_time = time.time()
        #print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        #print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(x_trn.shape[0]))
        t_ng_start = time.time()  # naive greedy start time
        #subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            '''subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = self.grads_per_elem[subset_selected]
            gains = self.eval_taylor_modular(rem_grads, greedySet, subset_selected)'''
            gains = self.eval_taylor_modular(self.grads_per_elem[remainSet])#rem_grads)
            # Update the greedy set and remaining set
            #bestId = subset_selected[torch.argmax(gains).item()]
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
        return list(greedySet)

class SetFunctionFacLoc(object):

    def __init__(self, device, batch_size):#, valid_loader):
        
        self.batch_size = batch_size   
        self.device = device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def distance(self,x, y, exp = 2):

      n = x.size(0)
      m = y.size(0)
      d = x.size(1)

      #print(x)
      #print(x.shape)
      #print(y.shape)
      #print("n="+str(n)+" m="+str(m)+" d="+str(d))
      x = x.unsqueeze(1).expand(n, m, d)
      y = y.unsqueeze(0).expand(n, m, d)

      dist = torch.pow(x - y, exp).sum(2) 
      return dist 

    def compute_score(self, x_trn, model):
      
      self.N = x_trn.shape[0]

      self.sim_mat = torch.zeros([self.N, self.N],dtype=torch.float32)
      
      for item in range(math.ceil(len(x_trn) / self.batch_size)):
      	
      	g_i = x_trn[ item*self.batch_size : (item + 1)*self.batch_size ]
      	for j in range(math.ceil(len(x_trn) / self.batch_size)):
      		
      		self.sim_mat[item*self.batch_size : (item + 1)*self.batch_size , j*self.batch_size : (j + 1)*self.batch_size] = self.distance(g_i, x_trn[ j*self.batch_size : (j + 1)*self.batch_size ])
      
      self.const = torch.max(self.sim_mat).item()
      self.sim_mat = self.const - self.sim_mat
      #self.sim_mat = self.sim_mat.to(self.device)
      dist = self.sim_mat.sum(1)
      bestId = torch.argmax(dist).item()
      self.max_sim = self.sim_mat[bestId].to(self.device)
      return bestId


    def lazy_greedy_max(self, budget, x_trn, model):
      
      id_first = self.compute_score(x_trn,model)
      self.gains = PriorityQueue()
      for i in range(self.N):
        if i == id_first :
          continue
        curr_gain = (torch.max(self.max_sim ,self.sim_mat[i].to(self.device)) - self.max_sim).sum()
        self.gains.put((-curr_gain.item(),i))

      numSelected = 2
      second = self.gains.get()
      greedyList = [id_first, second[1]]
      self.max_sim = torch.max(self.max_sim,self.sim_mat[second[1]].to(self.device))


      while(numSelected < budget):

          if self.gains.empty():
            break

          elif self.gains.qsize() == 1:
            bestId = self.gains.get()[1]

          else:
   
            bestGain = -np.inf
            bestId = None
            
            while True:

              first =  self.gains.get()

              if bestId == first[1]: 
                break

              curr_gain = (torch.max(self.max_sim, self.sim_mat[first[1]].to(self.device)) - self.max_sim).sum()
              self.gains.put((-curr_gain.item(), first[1]))


              if curr_gain.item() >= bestGain:
                  
                bestGain = curr_gain.item()
                bestId = first[1]

          greedyList.append(bestId)
          numSelected += 1

          self.max_sim = torch.max(self.max_sim,self.sim_mat[bestId].to(self.device))

      #print()
      #gamma = self.compute_gamma(greedyList)

      return greedyList

class SetFunctionBatch(object):
    
    def __init__(self, X_val, Y_val, model, loss_criterion, loss_nored, eta, device):
        
        self.x_val = X_val
        self.y_val = Y_val
        
        self.model = model
        self.loss = loss_criterion # For validation loss
        self.loss_nored = loss_nored # Make sure it has reduction='none' instead of default
        self.eta = eta # step size for the one step gradient update
        
        self.grads_per_elem = None
        self.device = device

    def _compute_per_element_grads(self, x_trn,y_trn,theta_init):
        
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        scores = self.model(x_trn)
        losses = self.loss_nored(scores, y_trn)
        self.N_trn = y_trn.shape[0]
        grads_vec = [0 for _ in range(self.N_trn)]   # zero is just a placeholder
        for item in range(self.N_trn):
            grads_vec[item] = torch.autograd.grad(losses[item], self.model.parameters(), retain_graph=True)
        self.grads_per_elem = grads_vec


    def _simple_eval(self, grads_X, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()):                                
                param.data.sub_(self.eta * grads_X[i])  
            scores = self.model(self.x_val)
            loss = self.loss(scores, self.y_val)
        return  -1.0 * loss.item()

    ## Computes the Validation Loss using the subset: X + elem by utilizing the 
    ## gradient of model parameters.
    def eval(self, grads_X, grads_elem, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()):                                
                param.data.sub_(self.eta * (grads_X[i] + grads_elem[i]))  
            scores = self.model(self.x_val)
            loss = self.loss(scores, self.y_val)
        return  -1.0 * loss.item()   

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the input e vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        for i, _ in enumerate(self.model.parameters()):
            grads_X[i] += grads_e[i]


    def naive_greedy_max(self, x_trn, y_trn, budget, theta_init):
        
        self._compute_per_element_grads(x_trn,y_trn, theta_init)
        #print("Computed train set gradients")
        numSelected = 0
        grads_currX = []   # basically stores grads_X for the current greedy set X
        greedySet = set()
        remainSet = set(range(self.N_trn))
        t_ng_start = time.time()    # naive greedy start time
        while(numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf # value for current iteration (validation loss)
            bestId = -1 # element to pick
            t_one_elem = time.time()
          
            for i in remainSet:
                grads_i = self.grads_per_elem[i]
                ## If no elements selected, use the self._simple_eval to get validation loss
                val_i = self.eval(grads_currX, grads_i, theta_init) if numSelected > 0 else self._simple_eval(grads_i ,theta_init)
                if val_i > bestGain:
                    bestGain = val_i
                    bestId = i

            # Update the greedy set and remaining set
            #print(bestGain,bestId)
            greedySet.add(bestId)
            remainSet.remove(bestId)    
            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:   # If 1st selection, then just set it to bestId grads
                grads_currX = list(self.grads_per_elem[bestId]) # Making it a list so that is mutable!                            
            if numSelected % 500 == 0:
                # Printing bestGain and Selection time for 1 element.
               print("numSelected:", numSelected, "Time for 1:", time.time()-t_one_elem, "bestGain:", bestGain)
            numSelected += 1
        print("Naive greedy total time:", time.time()-t_ng_start)
        return list(greedySet)
        # return greedySet, grads_currX
