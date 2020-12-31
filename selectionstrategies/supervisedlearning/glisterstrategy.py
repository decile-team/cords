import numpy as np
import time
import torch
import math
import torch.nn.functional as F
from selectionstrategies.supervisedlearning.dataselectionstrategy import DataSelectionStrategy


class GlisterStrategy(DataSelectionStrategy):

    def __init__(self, trainloader, valloader, model, loss_criterion,
                 eta, device, num_classes, linear_layer, selection_type):
        super().__init__(trainloader, valloader, model, linear_layer, selection_type)
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.eta = eta  # step size for the one step gradient update
        self.device = device
        self.num_classes = num_classes
        self.init_out = list()
        self.init_l1 = list()

    def _update_grads_val(self, grads_currX=None, first_init=False):
        self.model.zero_grad()
        embDim = self.model.get_embedding_dim()

        if first_init:
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    with torch.no_grad():
                        out, l1 = self.model(inputs, last=True)
                        data = F.softmax(out, dim=1)
                    #Gradient Calculation Part
                    outputs = torch.zeros(len(inputs), self.num_classes).to(self.device)
                    outputs.scatter_(1, targets.view(-1, 1), 1)
                    l0_grads = data - outputs
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    self.init_out = out
                    self.init_l1 = l1
                    self.y_val = targets.view(-1, 1)
                else:
                    with torch.no_grad():
                        out, l1 = self.model(inputs, last=True)
                        data = F.softmax(out, dim=1)
                    outputs = torch.zeros(len(inputs), self.num_classes).to(self.device)
                    outputs.scatter_(1, targets.view(-1, 1), 1)
                    batch_l0_grads = data - outputs
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
                    self.init_out = torch.cat((self.init_out, out), dim=0)
                    self.init_l1 = torch.cat((self.init_l1, l1), dim=0)
                    self.y_val = torch.cat((self.y_val, targets.view(-1, 1)), dim=0)

        elif grads_currX is not None:
            with torch.no_grad():
                out = torch.zeros(self.init_out.shape[0], self.init_out.shape[1]).to(self.device)
                for j in range(self.num_classes):
                    out[:, j] = self.init_out[:, j] - (1 * self.eta * (torch.matmul(self.init_l1, grads_currX[0][(j * embDim) +
                                self.num_classes:((j + 1) * embDim) + self.num_classes].view(-1, 1)) + grads_currX[0][j])).view(-1)
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                if self.linear_layer:
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
        torch.cuda.empty_cache()
        if self.linear_layer:
            self.grads_val_curr = torch.mean(torch.cat((l0_grads, l1_grads), dim=1), dim=0).view(-1, 1)
        else:
            self.grads_val_curr = torch.mean(l0_grads, dim=0).view(-1, 1)

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
    def select(self, budget, model_dict):
        self.update_model(model_dict)
        start_time = time.time()
        self.compute_gradients()
        end_time = time.time()
        print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        self.numSelected = 0
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        #if self.selection_type = 'R'
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (self.numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
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
            self._update_grads_val(grads_currX)
            if (self.numSelected - 1) % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX
