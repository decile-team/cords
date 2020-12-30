import numpy as np
import time
import torch


## One Step Set Functions on Validation Loss using just the last layer
class SetFunctionTaylorDeep(object):
    
    def __init__(self, X_trn, Y_trn, X_val, Y_val, valid, model, loss_criterion, loss_nored, eta,device):
        
        self.x_trn = X_trn
        self.y_trn = Y_trn
        if valid:
            self.x_val = X_val
            self.y_val = Y_val
        else:
            self.x_val = X_trn
            self.y_val = Y_trn
        self.model = model
        self.loss = loss_criterion # For validation loss
        self.loss_nored = loss_nored # Make sure it has reduction='none' instead of default
        self.eta = eta # step size for the one step gradient update
        self.N_trn = X_trn.shape[0]
        self.grads_per_elem = None
        self.first_element = True
        self.grads_val_curr = None
        self.device = device


    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        scores = self.model(self.x_trn)
        losses = self.loss_nored(scores, self.y_trn)
        N = self.N_trn
        grads_vec = [0 for _ in range(N)]   # zero is just a placeholder

        for name,_ in self.model.named_children():
            self.last_layer_name = name

        self.last_params =[]
        for name, param in self.model.named_parameters():
            if self.last_layer_name not in name :
                param.requires_grad = False
            else:
                self.last_params.append(param)

        for item in range(N):
            grads_vec[item] = torch.autograd.grad(losses[item], self.last_params, only_inputs=True, retain_graph=True)

        self.grads_per_elem = grads_vec


    def _compute_init_valloss_grads(self, theta_init):
        # Now compute the Validation loss initial gradient
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        scores = self.model(self.x_val)
        floss = -1 * self.loss(scores, self.y_val)
        self.grads_curr_subset  = torch.autograd.grad(floss, self.last_params, only_inputs=True)
        self.first_element = False


    def eval(self, grads_val, grads_elem, theta_init):
        return torch.dot(grads_val, -1 * self.eta * grads_elem)     
        

    def _update_gradients_subset(self, grads_X, element,theta_init):
        
        if not grads_X:
            grads_X = list(self.grads_per_elem[element])
        else:
            grads_e = self.grads_per_elem[element]
            for i, _ in enumerate(self.last_params):
                grads_X[i] += grads_e[i]

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.last_params):#self.model.parameters()):                                
                param.data.sub_(self.eta * grads_X[i])  
        # t = time.time()
        scores = self.model(self.x_val)
        floss = -1 * self.loss(scores, self.y_val)            
        self.grads_curr_subset = torch.autograd.grad(floss, self.last_params, only_inputs=True)
        return grads_X
        # e = time.time() - t
        # print("grad update time:", e)


    def flatten_params(self, param_list):
        l = [torch.flatten(p) for p in param_list]
        flat = torch.cat(l)
        return flat


    def naive_greedy_max(self, budget, theta_init):
        self._compute_per_element_grads(theta_init)
        self._compute_init_valloss_grads(theta_init)
        #print("Computed train set gradients")
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        #remainSet = set(range(self.N_trn))
        remainList = list(range(self.N_trn))
        t_ng_start = time.time()    # naive greedy start time
        flat_grads_val = self.flatten_params(self.grads_curr_subset).to(self.device) ## Total parameter size
        #flat_grads_val = flat_grads_val.to(self.device)
        num_params = flat_grads_val.shape[0]

        flat_grads_mat = torch.zeros((self.N_trn, num_params))
        for i in range(self.N_trn):
            flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])
        flat_grads_mat = flat_grads_mat.to(self.device)
        
        while(numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf # value for current iteration (validation loss)
            bestId = -1 # element to pick
            t_one_elem = time.time()

            #idxs_remain = list(remainSet)
            all_gains = torch.matmul(-1.0 * self.eta * flat_grads_mat[remainList], flat_grads_val)
            # print(all_gains.shape)
            tmpid = int(torch.argmax(all_gains))
            bestId = remainList[tmpid]
            bestGain = all_gains[tmpid]

            greedySet.add(bestId)
            remainList.remove(bestId)            
            grads_currX = self._update_gradients_subset(grads_currX, bestId, theta_init)    
            flat_grads_val = self.flatten_params(self.grads_curr_subset).to(self.device)                 
            
            numSelected += 1
        #print("Naive greedy total time with taylor:", time.time()-t_ng_start)

        return list(greedySet)#, grads_currX


class SetFunctionTaylor(object):
    
    def __init__(self, X_trn, Y_trn, X_val, Y_val, valid, model, 
            loss_criterion, loss_nored, eta,device):
        self.x_trn = X_trn
        self.y_trn = Y_trn
        if valid:
            self.x_val = X_val
            self.y_val = Y_val
        else:
            self.x_val = X_trn
            self.y_val = Y_trn
        self.model = model
        self.loss = loss_criterion # For validation loss
        self.loss_nored = loss_nored # Make sure it has reduction='none' instead of default
        self.eta = eta # step size for the one step gradient update
        self.N_trn = X_trn.shape[0]
        self.grads_per_elem = None
        self.first_element = True
        self.grads_val_curr = None
        self.device = device


    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        scores = self.model(self.x_trn)
        losses = self.loss_nored(scores, self.y_trn)
        N = self.N_trn
        grads_vec = [0 for _ in range(N)]   # zero is just a placeholder
        for item in range(N):
            grads_vec[item] = torch.autograd.grad(losses[item], self.model.parameters(), retain_graph=True)
        self.grads_per_elem = grads_vec


    def _compute_init_valloss_grads(self, theta_init):
        # Now compute the Validation loss initial gradient
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        scores = self.model(self.x_val)
        floss = -1 * self.loss(scores, self.y_val)
        self.grads_curr_subset  = torch.autograd.grad(floss, self.model.parameters())
        self.first_element = False


    # Updates gradients of set X + element (basically adding element to X)
    #Also computes F'(theta_X) part of taylor approx
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element, theta_init):
        if not grads_X:
            grads_X = list(self.grads_per_elem[element])
        else:
            grads_e = self.grads_per_elem[element]
            for i, _ in enumerate(self.model.parameters()):
                grads_X[i] += grads_e[i]

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()):                                
                param.data.sub_(self.eta * grads_X[i])  
        # t = time.time()
        scores = self.model(self.x_val)
        floss = -1 * self.loss(scores, self.y_val)            
        self.grads_curr_subset = torch.autograd.grad(floss, self.model.parameters())
        return grads_X


    def flatten_params(self, param_list):
        l = [torch.flatten(p) for p in param_list]
        flat = torch.cat(l)
        return flat


    def naive_greedy_max(self, budget, theta_init):
        self._compute_per_element_grads(theta_init)
        self._compute_init_valloss_grads(theta_init)
        print("Computed train set gradients")
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        remainSet = set(range(self.N_trn))
        remainList = list(range(self.N_trn))
        t_ng_start = time.time()    # naive greedy start time
        flat_grads_val = self.flatten_params(self.grads_curr_subset) ## Total parameter size
        num_params = flat_grads_val.shape[0]

        flat_grads_mat = torch.zeros((self.N_trn, num_params))
        for i in range(self.N_trn):
            flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])
        flat_grads_mat = flat_grads_mat.to(self.device)
        
        while(numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf # value for current iteration (validation loss)
            bestId = -1 # element to pick
            t_one_elem = time.time()
            flat_grads_val = self.flatten_params(self.grads_curr_subset)
            idxs_remain = list(remainSet)
            all_gains = torch.matmul(-1.0 * self.eta * flat_grads_mat[idxs_remain], flat_grads_val)
            # print(all_gains.shape)
            tmpid = int(torch.argmax(all_gains))
            bestId = idxs_remain[tmpid]
            bestGain = all_gains[tmpid]

            # Update the greedy set and remaining set
            greedySet.add(bestId)
            remainSet.remove(bestId)            
            grads_currX = self._update_gradients_subset(grads_currX, bestId, theta_init)                           
            # print("numSelected:", numSelected, "Time for 1:", time.time()-t_one_elem, "bestGain:", bestGain)
            if numSelected % 500 == 0:
                # Printing bestGain and Selection time for 1 element.
               print("numSelected:", numSelected, "Time for 1:", time.time()-t_one_elem, "bestGain:", bestGain)
            numSelected += 1
        print("Naive greedy total time with taylor:", time.time()-t_ng_start)

        return list(greedySet)


class SetFunctionBatch(object):
    
    def __init__(self, X_trn, Y_trn, X_val, Y_val, valid, model, 
            loss_criterion, loss_nored, eta, device):
        self.x_trn = X_trn
        self.y_trn = Y_trn
        if valid:   # Use Validation Data to do the selection
            self.x_val = X_val
            self.y_val = Y_val
        else:       # Use Training Data to do the selection
            self.x_val = X_trn
            self.y_val = Y_trn
        self.model = model
        self.loss = loss_criterion # For validation loss
        self.loss_nored = loss_nored # Make sure it has reduction='none' instead of default
        self.eta = eta # step size for the one step gradient update
        self.N_trn = X_trn.shape[0]
        self.grads_per_elem = None
        self.device = device

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        scores = self.model(self.x_trn)
        losses = self.loss_nored(scores, self.y_trn)
        N = self.N_trn
        grads_vec = [0 for _ in range(N)]   # zero is just a placeholder
        for item in range(N):
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
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        for i, _ in enumerate(self.model.parameters()):
            grads_X[i] += grads_e[i]


    def naive_greedy_max(self, budget, theta_init):
        self._compute_per_element_grads(theta_init)
        print("Computed train set gradients")
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





## Comparison Set Function to see how the gains look and what's the rank.
class SetFunctionCompare(object):
    
    def __init__(self, X_trn, Y_trn, X_val, Y_val, valid,model, 
            loss_criterion, loss_nored, eta,device):
        self.x_trn = X_trn
        self.y_trn = Y_trn
        if valid:
            self.x_val = X_val
            self.y_val = Y_val
        else:
            self.x_val = X_trn
            self.y_val = Y_trn
        self.model = model
        self.loss = loss_criterion # For validation loss
        self.loss_nored = loss_nored # Make sure it has reduction='none' instead of default
        self.eta = eta # step size for the one step gradient update
        self.N_trn = X_trn.shape[0]
        self.grads_per_elem = None
        self.first_element = True
        self.grads_val_curr = None
        self.device = device


    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        scores = self.model(self.x_trn)
        losses = self.loss_nored(scores, self.y_trn)
        N = self.N_trn
        grads_vec = [0 for _ in range(N)]   # zero is just a placeholder
        for item in range(N):
            grads_vec[item] = torch.autograd.grad(losses[item], self.model.parameters(), retain_graph=True)
        self.grads_per_elem = grads_vec

    ## Non-tay version
    def _simple_eval(self, grads_X, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()):                                
                param.data.sub_(self.eta * grads_X[i])  
            scores = self.model(self.x_val)
            loss = self.loss(scores, self.y_val)
        return  -1.0 * loss.item()

    ## Non-tay version
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


    def _compute_init_valloss_grads(self, theta_init):
        # Now compute the Validation loss initial gradient
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        scores = self.model(self.x_val)
        floss = -1 * self.loss(scores, self.y_val)
        self.grads_curr_subset  = torch.autograd.grad(floss, self.model.parameters())
        self.first_element = False

    '''
    ## Computes the Validation Loss using the subset: X + elem by utilizing the 
    ## gradient of model parameters.
    def eval(self, grads_val, grads_elem, theta_init):
        return torch.dot(grads_val, -1 * self.eta * grads_elem)     
        # dot_prod = 0
        # # t = time.time()
        # # self.model.load_state_dict(theta_init) # Model not really needed here!
        # with torch.no_grad():
        #     for i, param in enumerate(self.model.parameters()):
        #         dot_prod += torch.sum(self.grads_curr_subset[i] * (- self.eta * grads_elem[i]))
        # # e = time.time() - t
        # # print("dotprod time:", e)
        # print(dp1, dot_prod)
        # return dot_prod.data
    '''

    def _old_update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        for i, _ in enumerate(self.model.parameters()):
            grads_X[i] += grads_e[i]



    # Updates gradients of set X + element (basically adding element to X)
    #Also computes F'(theta_X) part of taylor approx
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element,theta_init):
        if not grads_X:
            grads_X = list(self.grads_per_elem[element])
        else:
            grads_e = self.grads_per_elem[element]
            for i, _ in enumerate(self.model.parameters()):
                grads_X[i] += grads_e[i]

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()):                                
                param.data.sub_(self.eta * grads_X[i])  
        # t = time.time()
        scores = self.model(self.x_val)
        floss = -1 * self.loss(scores, self.y_val)            
        self.grads_curr_subset = torch.autograd.grad(floss, self.model.parameters())
        return grads_X
        # e = time.time() - t
        # print("grad update time:", e)


    def flatten_params(self, param_list):
        l = [torch.flatten(p) for p in param_list]
        flat = torch.cat(l)
        return flat


    def naive_greedy_max(self, budget, theta_init):
        self._compute_per_element_grads(theta_init)
        self._compute_init_valloss_grads(theta_init)
        # print("Computed train set gradients")
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        remainSet = set(range(self.N_trn))
        remainList = list(range(self.N_trn))

        old_grads_currX = []   # basically stores grads_X for the current greedy set X
        old_greedySet = set()
        old_remainSet = set(range(self.N_trn))

        t_ng_start = time.time()    # naive greedy start time
        flat_grads_val = self.flatten_params(self.grads_curr_subset) ## Total parameter size
        num_params = flat_grads_val.shape[0]

        flat_grads_mat = torch.zeros((self.N_trn, num_params))
        for i in range(self.N_trn):
            flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])
        flat_grads_mat = flat_grads_mat.to(self.device)
        
        l2_diff_all = torch.zeros(budget)
        l1_diff_all = torch.zeros(budget)
        rel_l1 = torch.zeros(budget)
        ranks_all = torch.zeros(budget)

        while(numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf # value for current iteration (validation loss)
            bestId = -1 # element to pick

            old_bestGain = -np.inf
            old_bestId = -1
            
            t_one_elem = time.time()
            flat_grads_val = self.flatten_params(self.grads_curr_subset)
            idxs_remain = list(remainSet)
            curr_gains = torch.matmul(-1.0 * self.eta * flat_grads_mat[idxs_remain], flat_grads_val)
            all_gains = torch.matmul(-1.0 * self.eta * flat_grads_mat, flat_grads_val)
            
            # print(all_gains.shape)
            tmpid = int(torch.argmax(curr_gains))
            bestId = idxs_remain[tmpid]
            bestGain = curr_gains[tmpid]

            # Update the greedy set and remaining set
            greedySet.add(bestId)
            remainSet.remove(bestId)            
            grads_currX = self._update_gradients_subset(grads_currX, bestId, theta_init)                           

            for i in old_remainSet:
                grads_i = self.grads_per_elem[i]
                ## If no elements selected, use the self._simple_eval to get validation loss
                val_i = self.eval(old_grads_currX, grads_i, theta_init) if numSelected > 0 else self._simple_eval(grads_i ,theta_init)
                if val_i > old_bestGain:
                    old_bestGain = val_i
                    old_bestId = i
            old_greedySet.add(old_bestId)
            old_remainSet.remove(old_bestId)            

            if numSelected > 0:
                old_actual_gain = old_bestGain - self._simple_eval(old_grads_currX, theta_init)
                self._old_update_gradients_subset(old_grads_currX, bestId)
            else:   # If 1st selection, then just set it to bestId grads
                dummy_grads = [0 for i in self.model.parameters()]
                old_actual_gain = old_bestGain - self._simple_eval(dummy_grads, theta_init)
                old_grads_currX = list(self.grads_per_elem[old_bestId]) # Making it a list so that is mutable!                            
            
            
            diff = old_actual_gain - all_gains[old_bestId]
            l1_diff_all[numSelected] = torch.abs(diff)
            l2_diff_all[numSelected] = diff**2

            rel_l1_diff = torch.abs(diff) / old_actual_gain
            rel_l1[numSelected] = rel_l1_diff

            sorted_taylor_gains = torch.argsort(all_gains, descending=True)
            list_sorted = [i.item() for i in sorted_taylor_gains]
            rank_oldbestId = list_sorted.index(old_bestId)
            ranks_all[numSelected] = rank_oldbestId

            print("Numselected:", numSelected)
            print("Current Diff:", diff, old_actual_gain, all_gains[old_bestId])
            print("curr abs diff:", torch.abs(diff))
            print("curr relative L1 diff:", rel_l1_diff)
            print("curr squared diff:", diff**2)
            print("curr rank:", rank_oldbestId)
            print("Old method best gain, bestId:", old_actual_gain, old_bestId)
            print("Tay method best gain, bestId:", bestGain, bestId)                


            # if numSelected % 500 == 0:
            #     # Printing bestGain and Selection time for 1 element.
            #    print("numSelected:", numSelected, "Time for 1:", time.time()-t_one_elem, "bestGain:", bestGain)
            numSelected += 1
        # print("Naive greedy total time with taylor:", time.time()-t_ng_start)
        print("\n")        
        print("--------selection over-----------")
        print("L2 Diff Mean: ", l2_diff_all.mean())
        print("L1 Diff Mean: ", l1_diff_all.mean())
        print("Avg Rel L1:", rel_l1.mean())
        print("Average of ranks: ", ranks_all.mean())

        return list(greedySet), grads_currX




class SetFunctionTaylorDebug(object):
    
    def __init__(self, X_trn, Y_trn, X_val, Y_val, valid, model, 
            loss_criterion, loss_nored, eta,device):
        self.x_trn = X_trn
        self.y_trn = Y_trn
        if valid:
            self.x_val = X_val
            self.y_val = Y_val
        else:
            self.x_val = X_trn
            self.y_val = Y_trn
        self.model = model
        self.loss = loss_criterion # For validation loss
        self.loss_nored = loss_nored # Make sure it has reduction='none' instead of default
        self.eta = eta # step size for the one step gradient update
        self.N_trn = X_trn.shape[0]
        self.grads_per_elem = None
        self.first_element = True
        self.grads_val_curr = None
        self.device = device


    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        scores = self.model(self.x_trn)
        losses = self.loss_nored(scores, self.y_trn)
        N = self.N_trn
        grads_vec = [0 for _ in range(N)]   # zero is just a placeholder
        for item in range(N):
            grads_vec[item] = torch.autograd.grad(losses[item], self.model.parameters(), retain_graph=True)
        self.grads_per_elem = grads_vec


    def _compute_init_valloss_grads(self, theta_init):
        # Now compute the Validation loss initial gradient
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        scores = self.model(self.x_val)
        floss = -1 * self.loss(scores, self.y_val)        
        self.grads_curr_subset  = torch.autograd.grad(floss, self.model.parameters())
        # print("INIT LOSS", -1 * floss.item())
        self.first_element = False


    # Updates gradients of set X + element (basically adding element to X)
    #Also computes F'(theta_X) part of taylor approx
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element, theta_init):
        if not grads_X:
            grads_X = list(self.grads_per_elem[element])
        else:
            grads_e = self.grads_per_elem[element]
            for i, _ in enumerate(self.model.parameters()):
                grads_X[i] += grads_e[i]

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()):                                
                param.data.sub_(self.eta * grads_X[i])  
        # t = time.time()
        scores = self.model(self.x_val)
        floss = -1 * self.loss(scores, self.y_val)        
        self.grads_curr_subset = torch.autograd.grad(floss, self.model.parameters())
        # print(-1 * floss.item())
        return grads_X


    def flatten_params(self, param_list):
        l = [torch.flatten(p) for p in param_list]
        flat = torch.cat(l)
        return flat


    def naive_greedy_max(self, budget, theta_init):
        self._compute_per_element_grads(theta_init)
        self._compute_init_valloss_grads(theta_init)
        print("Computed train set gradients")
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        remainSet = set(range(self.N_trn))
        remainList = list(range(self.N_trn))
        t_ng_start = time.time()    # naive greedy start time
        flat_grads_val = self.flatten_params(self.grads_curr_subset) ## Total parameter size
        num_params = flat_grads_val.shape[0]

        flat_grads_mat = torch.zeros((self.N_trn, num_params))
        for i in range(self.N_trn):
            flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])
        flat_grads_mat = flat_grads_mat.to(self.device)
        
        while(numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf # value for current iteration (validation loss)
            bestId = -1 # element to pick
            t_one_elem = time.time()
            flat_grads_val = self.flatten_params(self.grads_curr_subset)
            idxs_remain = list(remainSet)
            all_gains = torch.matmul(-1.0 * self.eta * flat_grads_mat[idxs_remain], flat_grads_val)
            # print(all_gains.shape)
            tmpid = int(torch.argmax(all_gains))
            # tmpid = int(torch.argmax(all_gains))
            bestId = idxs_remain[tmpid] # out of the remaining indices, which is the best?
            bestGain = all_gains[tmpid]

            # Update the greedy set and remaining set
            greedySet.add(bestId)
            remainSet.remove(bestId)            
            grads_currX = self._update_gradients_subset(grads_currX, bestId, theta_init)                           
            # print("numSelected:", numSelected, "Time for 1:", time.time()-t_one_elem, "bestGain:", bestGain)
            if numSelected % 500 == 0:
                # Printing bestGain and Selection time for 1 element.
               print("numSelected:", numSelected, "Time for 1:", time.time()-t_one_elem, "bestGain:", bestGain)
            numSelected += 1
        print("Naive greedy total time with taylor:", time.time()-t_ng_start)

        print("budget, selected:", budget, len(greedySet), self.N_trn, len(remainSet))

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()):                                
                param.data.sub_(self.eta * grads_currX[i])  
        # t = time.time()
        scores = self.model(self.x_val)
        floss = -1 * self.loss(scores, self.y_val)        
        self.grads_curr_subset = torch.autograd.grad(floss, self.model.parameters())
        print("AFTER SELECTION VAL LOSS:", -1 * floss.item())

        classes, counts = np.unique(self.y_trn, return_counts=True)

        counts_set = np.zeros(len(classes))
        for idx in greedySet:
            y_lab = self.y_trn[idx]
            counts_set[y_lab] += 1


        # Random Subset
        state = np.random.get_state()
        np.random.seed(101)
        myset = np.random.choice(self.N_trn, size=budget, replace=False)
        np.random.set_state(state)

        counts_rand = np.zeros(len(classes))
        for idx in myset:
            y_lab = self.y_trn[idx]
            counts_rand[y_lab] += 1


        print("True TrnD dist:", counts/np.sum(counts))
        print("greedySet dist:", counts_set/np.sum(counts_set))
        print("RandSet --dist:", counts_rand/np.sum(counts_rand))

        return list(greedySet), grads_currX       

        # return myset 
        # return np.random.choice(self.N_trn, size=budget, replace=False)

    def naive_greedy_max_perclass(self, budget, theta_init):
        self._compute_per_element_grads(theta_init)
        print("Computed train set gradients")
        global_greedySet = set()

        classes, counts = np.unique(self.y_trn, return_counts=True)
        class_dist = counts/np.sum(counts)
        self._compute_init_valloss_grads(theta_init)
        flat_grads_val = self.flatten_params(self.grads_curr_subset) ## Total parameter size
        num_params = flat_grads_val.shape[0]
        flat_grads_mat = torch.zeros((self.N_trn, num_params))
        
        for i in range(self.N_trn):
            flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])
        flat_grads_mat = flat_grads_mat.to(self.device)
        
        for curr_cls in classes:
            greedySet = set()
            mask_curr_cls = (self.y_trn == curr_cls)
            # idxs_curr_cls = list()
            remainSet = set(range(self.N_trn))
            
            numSelected = 0
            grads_currX = []  # basically stores grads_X for the current greedy set X
            greedySet = set()
            
        



        while(numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf # value for current iteration (validation loss)
            bestId = -1 # element to pick
            t_one_elem = time.time()
            flat_grads_val = self.flatten_params(self.grads_curr_subset)
            idxs_remain = list(remainSet)
            all_gains = torch.matmul(-1.0 * self.eta * flat_grads_mat[idxs_remain], flat_grads_val)
            # print(all_gains.shape)
            tmpid = int(torch.argmax(all_gains))
            # tmpid = int(torch.argmax(all_gains))
            bestId = idxs_remain[tmpid] # out of the remaining indices, which is the best?
            bestGain = all_gains[tmpid]

            # Update the greedy set and remaining set
            greedySet.add(bestId)
            remainSet.remove(bestId)            
            grads_currX = self._update_gradients_subset(grads_currX, bestId, theta_init)                           
            # print("numSelected:", numSelected, "Time for 1:", time.time()-t_one_elem, "bestGain:", bestGain)
            if numSelected % 500 == 0:
                # Printing bestGain and Selection time for 1 element.
               print("numSelected:", numSelected, "Time for 1:", time.time()-t_one_elem, "bestGain:", bestGain)
            numSelected += 1
        print("Naive greedy total time with taylor:", time.time()-t_ng_start)

        print("budget, selected:", budget, len(greedySet), self.N_trn, len(remainSet))

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()):                                
                param.data.sub_(self.eta * grads_currX[i])  
        # t = time.time()
        scores = self.model(self.x_val)
        floss = -1 * self.loss(scores, self.y_val)        
        self.grads_curr_subset = torch.autograd.grad(floss, self.model.parameters())
        print("AFTER SELECTION VAL LOSS:", -1 * floss.item())

        

        counts_set = np.zeros(len(classes))
        for idx in greedySet:
            y_lab = self.y_trn[idx]
            counts_set[y_lab] += 1


        # Random Subset
        state = np.random.get_state()
        np.random.seed(101)
        myset = np.random.choice(self.N_trn, size=budget, replace=False)
        np.random.set_state(state)

        counts_rand = np.zeros(len(classes))
        for idx in myset:
            y_lab = self.y_trn[idx]
            counts_rand[y_lab] += 1


        print("True TrnD dist:", counts/np.sum(counts))
        print("greedySet dist:", counts_set/np.sum(counts_set))
        print("RandSet --dist:", counts_rand/np.sum(counts_rand))

        return list(greedySet), grads_currX       

        # return myset 
        # return np.random.choice(self.N_trn, size=budget, replace=False)



