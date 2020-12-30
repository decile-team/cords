import numpy as np
import time
import torch
from queue import PriorityQueue  # lazy greedy
import sys


## One Step Set Functions on Validation Loss
## Batch and Pytorch Dataloader variants implemented.


class SetFunctionBatch(object):

    def __init__(self, X_trn, Y_trn, X_val, Y_val, model,
                 loss_criterion, loss_nored, eta):
        self.x_trn = X_trn
        self.x_val = X_val
        self.y_trn = Y_trn
        self.y_val = Y_val
        self.model = model
        self.loss = loss_criterion  # For validation loss
        self.loss_nored = loss_nored  # Make sure it has reduction='none' instead of default
        self.eta = eta  # step size for the one step gradient update
        self.N_trn = X_trn.shape[0]
        self.grads_per_elem = None

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        scores = self.model(self.x_trn)
        losses = self.loss_nored(scores, self.y_trn)
        N = self.N_trn
        grads_vec = [0 for _ in range(N)]  # zero is just a placeholder
        for item in range(N):
            grads_vec[item] = torch.autograd.grad(losses[item], self.model.parameters(), retain_graph=True)
        self.grads_per_elem = grads_vec

    def _simple_eval(self, grads_X, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad():  # perform one-step update
            for i, param in enumerate(self.model.parameters()):
                param.data.sub_(self.eta * grads_X[i])
            scores = self.model(self.x_val)
            loss = self.loss(scores, self.y_val)
        return -1.0 * loss.item()

    ## Computes the Validation Loss using the subset: X + elem by utilizing the
    ## gradient of model parameters.
    def eval(self, grads_X, grads_elem, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad():  # perform one-step update
            for i, param in enumerate(self.model.parameters()):
                if grads_X:  # non empty grads_X
                    param.data.sub_(self.eta * (grads_X[i] + grads_elem[i]))
                else:
                    param.data.sub_(self.eta * grads_elem[i])
            scores = self.model(self.x_val)
            loss = self.loss(scores, self.y_val)
        return -1.0 * loss.item()

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the input vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        for i, _ in enumerate(self.model.parameters()):
            grads_X[i] += grads_e[i]

    def naive_greedy_max(self, budget, theta_init):
        self._compute_per_element_grads(theta_init)
        print("Computed train set gradients")
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        remainSet = set(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        while (numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf  # value for current iteration (validation loss)
            bestId = -1  # element to pick
            t_one_elem = time.time()

            for i in remainSet:
                grads_i = self.grads_per_elem[i]
                ## If no elements selected, use the self._simple_eval to get validation loss
                val_i = self.eval(grads_currX, grads_i, theta_init) if numSelected > 0 else self._simple_eval(grads_i,
                                                                                                              theta_init)
                if val_i > bestGain:
                    bestGain = val_i
                    bestId = i

            # print(-1 * bestGain)
            # Update the greedy set and remaining set
            greedySet.add(bestId)
            remainSet.remove(bestId)
            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = list(
                    self.grads_per_elem[bestId])  # Making it a list so that is mutable!
            if numSelected % 500 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem, "bestGain:", bestGain)
            numSelected += 1
        print("Naive greedy total time:", time.time() - t_ng_start)

        # Do an update to model parameters based on grads from greedy subset
        self.model.load_state_dict(theta_init)
        with torch.no_grad():  # perform one-step update
            for i, param in enumerate(self.model.parameters()):
                param.data.sub_(self.eta * grads_currX[i])
            scores = self.model(self.x_val)
            loss = self.loss(scores, self.y_val)
            print("final val loss:", loss)

        return list(greedySet), grads_currX
        # return greedySet, grads_currX

    def lazy_greedy_max(self, budget, theta_init):
        self._compute_per_element_grads(theta_init)
        print("Computed train set gradients")
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        remainSet = set(range(self.N_trn))
        t_lg_start = time.time()  # naive greedy start time
        pq = PriorityQueue()

        # push initial gains of every element to the queue.
        for i in remainSet:
            grads_i = self.grads_per_elem[i]
            val_i = self._simple_eval(grads_i, theta_init)
            pq.put((val_i, i))

        sort_cnt = 0
        while (not pq.empty()):
            topid = pq.get()[1]  # we want the id of top variable
            grads_topid = self.grads_per_elem[topid]
            gain_topid = self.eval(grads_currX, grads_topid, theta_init) if numSelected > 0 else self._simple_eval(
                grads_topid, theta_init)

            first_tup = pq.get()  # pop's the first element
            pq.put(first_tup)  # dont pop it! just need access to the gain value, so put it back
            if gain_topid < first_tup[0]:
                sort_cnt += 1
                pq.put((gain_topid, topid))
                print(sort_cnt)
            else:  # guranteed to be optimal
                greedySet.add(topid)
                remainSet.remove(topid)
                # update the "preCompute" for current greedySet -- sum of the gradients.
                if numSelected > 0:
                    self._update_gradients_subset(grads_currX, topid)
                else:  # If 1st selection, then just set it to bestId grads
                    grads_currX = list(
                        self.grads_per_elem[topid])  # Making it a list so that is mutable!

                numSelected += 1
                sort_cnt += 1
                if numSelected > budget:
                    break

        print("Lazy greedy total time:", time.time() - t_lg_start)
        return list(greedySet), grads_currX
        # return greedySet, grads_currX

    def lazy_greedy_max_2(self, budget, theta_init):
        self._compute_per_element_grads(theta_init)
        print("Computed train set gradients")
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        remainSet = set(range(self.N_trn))
        t_lg_start = time.time()  # naive greedy start time
        pq = PriorityQueue()

        # push initial gains of every element to the queue.
        for i in remainSet:
            grads_i = self.grads_per_elem[i]
            val_i = self._simple_eval(grads_i, theta_init)
            pq.put((val_i, i))

        sort_cnt = 0
        while (numSelected < budget):
            if pq.empty():
                break
            elif pq.qsize() == 1:
                bestId = pq.get()[1]
            else:
                bestId = None
                bestGain = -np.inf
                while True:
                    first = pq.get()
                    if bestId == first[1]:
                        break
                    grads_topid = self.grads_per_elem[first[1]]
                    cur_gain = self.eval(grads_currX, grads_topid,
                                         theta_init) if numSelected > 0 else self._simple_eval(grads_topid, theta_init)
                    pq.put((cur_gain, first[1]))
                    if cur_gain > bestGain:
                        bestGain = cur_gain
                        bestId = first[1]
            greedySet.add(bestId)
            remainSet.remove(bestId)
            # update the "preCompute" for current greedySet -- sum of the gradients.
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = list(
                    self.grads_per_elem[bestId])  # Making it a list so that is mutable!
            numSelected += 1

        print("Lazy greedy total time:", time.time() - t_lg_start)
        return list(greedySet), grads_currX
        # return greedySet, grads_currX


class SetFunctionLoader(object):
    def __init__(self, trainloader, valloader, model, loss_criterion,
                 loss_nored, eta, device):
        self.trainloader = trainloader  # assume its a sequential loader.
        self.valloader = valloader  # assume its a sequential loader.
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = len(trainloader.sampler)
        self.grads_per_elem = None

    def _compute_per_element_grads(self, theta_init):
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

    # Computes the validation loss using the vector of gradient sums in set X.
    def _simple_eval(self, grads_X, theta_init):
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

    ## Computes the Validation Loss using the subset: X + elem by utilizing the
    ## gradient of model parameters.
    def eval(self, grads_X, grads_elem, theta_init):
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

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        for i, _ in enumerate(self.model.parameters()):
            grads_X[i] += grads_e[i]

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init):
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


# Validation data is loaded in batch rather than a loader.
class SetFunctionLoader_2(object):

    def __init__(self, trainloader, validset, model, loss_criterion,
                 loss_nored, eta, device):
        self.trainloader = trainloader  # assume its a sequential loader.
        # self.valloader = valloader      # assume its a sequential loader.
        self.x_val = validset.dataset.data[validset.indices].to(device, dtype=torch.float).view(len(validset.indices),
                                                                                                1, 28, 28)
        self.y_val = validset.dataset.targets[validset.indices].to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = len(trainloader.sampler)
        self.grads_per_elem = None

    def _compute_per_element_grads(self, theta_init):
        grads_vec = [0 for i in range(self.N_trn)]
        counter = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # print(batch_idx)
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            self.model.load_state_dict(theta_init)
            scores = self.model(inputs)
            losses = self.loss_nored(scores, targets)
            for i, loss_i in enumerate(losses):
                self.model.zero_grad()
                grads_vec[i + counter] = [torch.autograd.grad(loss_i, self.model.parameters(), retain_graph=True)[-1]]
            params = [param for param in self.model.parameters()]
            for param in params:
                param.grad = None
                param.detach()
            torch.cuda.empty_cache()
            counter += len(targets)
            # if (batch_idx % 100 == 0):
            # print(torch.cuda.memory_allocated())
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = grads_vec

    def _simple_eval(self, grads_X, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad():  # perform one-step update
            params = [param for param in self.model.parameters()]
            params[-1].data.sub_(self.eta * grads_X[0])
            scores = self.model(self.x_val)
            loss = self.loss(scores, self.y_val)
        return -1.0 * loss.item()

    ## Computes the Validation Loss using the subset: X + elem by utilizing the
    ## gradient of model parameters.
    def eval(self, grads_X, grads_elem, theta_init):
        # t=time.time()
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad():  # perform one-step update
            params = [param for param in self.model.parameters()]
            params[-1].data.sub_(self.eta * (grads_X[0] + grads_elem[0]))
            scores = self.model(self.x_val)
            loss = self.loss(scores, self.y_val)
        # print('eval time:', time.time()-t)
        return -1.0 * loss.item()

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        # for i, _ in enumerate(self.model.parameters()):
        grads_X[0] += grads_e[0]

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init):
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



