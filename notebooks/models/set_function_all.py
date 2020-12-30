import math
import numpy as np
import time
import torch
from queue import PriorityQueue


class SetFunctionFacLoc(object):

    def __init__(self, device, train_full_loader):#, valid_loader):
        
        self.train_loader = train_full_loader      
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

    def compute_score(self, model):
      self.N = 0
      g_is =[]
      with torch.no_grad():
        for i, data_i in  enumerate(self.train_loader, 0):
          inputs_i, target_i = data_i
          inputs_i = inputs_i.to(self.device) #, target_i.to(self.device)
          self.N += inputs_i.size()[0]
          g_is.append(inputs_i)
        
        self.sim_mat = torch.zeros([self.N, self.N],dtype=torch.float32)

        first_i = True

        for i, g_i in enumerate(g_is, 0):

          if first_i:
            size_b = g_i.size(0)
            first_i = False

          for j, g_j in enumerate(g_is, 0):
            self.sim_mat[i*size_b: i*size_b + g_i.size(0), j*size_b: j*size_b + g_j.size(0)] = self.distance(g_i, g_j)
      self.const = torch.max(self.sim_mat).item()
      self.sim_mat = self.const - self.sim_mat
      #self.sim_mat = self.sim_mat.to(self.device)
      dist = self.sim_mat.sum(1)
      bestId = torch.argmax(dist).item()
      self.max_sim = self.sim_mat[bestId].to(self.device)
      return bestId


    def lazy_greedy_max(self, budget, model):
      id_first = self.compute_score(model)
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


class SetFunctionTaylorDeep_SuperRep(object):
    def __init__(self, x_trn, y_trn, trn_batch, x_val, y_val,valid, model, loss_criterion, loss_nored, eta,device,N_trn):
        self.x_trn = x_trn
        self.y_trn = y_trn
        self.trn_batch = trn_batch

        self.valid = valid
        self.x_val = x_val
        self.y_val = y_val

        self.model = model
        self.loss = loss_criterion # For validation loss
        self.loss_nored = loss_nored # Make sure it has reduction='none' instead of default
        self.eta = eta # step size for the one step gradient update
        self.N_trn = N_trn
        self.grads_per_elem = None
        self.first_element = True
        self.grads_val_curr = None
        self.device = device

        #print(len(self.valid_loader))
        #print(len(next(iter(self.valid_loader))[0]))
        #print(next(iter(self.valid_loader))[0][0])

    def class_wise(self,bud,theta_init):

      classes = torch.unique(self.y_trn)      

      self.N = self.y_trn.shape[0]
      greedyList =[]

      for cl in classes:

        idx = (self.y_trn == cl).nonzero().flatten()
        idx.tolist()

        curr_x_trn = self.x_trn[idx]
        curr_y_trn = self.y_trn[idx]
        self.curr_N = curr_y_trn.shape[0]
        #self.curr_bud = math.ceil(bud*self.curr_N / self.N)

        self.train_loader =[]
        for item in range(math.ceil(self.curr_N /self.trn_batch)):
          inputs = curr_x_trn[item*self.trn_batch:(item+1)*self.trn_batch]
          target  = curr_y_trn[item*self.trn_batch:(item+1)*self.trn_batch]
          self.train_loader.append((inputs,target))

        if self.valid:
          ind = (self.y_val == cl).nonzero().flatten()
          ind.tolist()

          curr_x_val = self.x_val[ind]
          curr_y_val = self.y_val[ind]
          self.curr_val_N = len(curr_y_val)

          self.valid_loader =[]
          for item in range(math.ceil(len(curr_y_val) /self.trn_batch)): #curr_y_val.shape[0]
            inputs = curr_x_val[item*self.trn_batch:(item+1)*self.trn_batch]
            target  = curr_y_val[item*self.trn_batch:(item+1)*self.trn_batch]
            self.valid_loader.append((inputs,target))

        else:
          self.curr_val_N = self.curr_N
          self.valid_loader = self.train_loader
        
        subset = self.naive_greedy_max(math.ceil(bud*self.curr_N / self.N), theta_init)

        for j in range(len(subset)):
          greedyList.append(idx[subset[j]])

      greedyList.sort()
      return greedyList


    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        losses =[]
        for i, data_i in  enumerate(self.train_loader, 0):
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          losses.append(self.loss_nored(scores, target_i))

        losses = torch.cat(losses,0)

        N = self.curr_N
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
            grads_vec[item] = torch.autograd.grad(losses[item],self.last_params , retain_graph=True) #self.model.parameters()

        self.grads_per_elem = grads_vec


    def _compute_init_valloss_grads(self, theta_init):
        # Now compute the Validation loss initial gradient
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        floss = 0
        #self.val_losses =[]
        for i, data_i in  enumerate(self.valid_loader, 0):
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          #self.val_losses.append(self.loss_nored(scores, target_i))
          floss += -1 * self.loss(scores, target_i)

        #self.val_losses = torch.cat(self.val_losses,0)
        
        self.grads_curr_subset  = torch.autograd.grad(floss/(i+1), self.last_params)#self.model.parameters())
        self.first_element = False    
        

    def _update_gradients_subset(self, grads_X, element, theta_init):
        
        if not grads_X:
            grads_X = list(self.grads_per_elem[element])
        else:
            grads_e = self.grads_per_elem[element]
            for i, _ in enumerate(self.last_params):#self.model.parameters()):
                grads_X[i] += grads_e[i]

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.last_params):#self.model.parameters()):                                
                param.data.sub_(self.eta * grads_X[i])  
        # t = time.time()
        floss = 0
        self.val_losses = []
        for i, data_i in  enumerate(self.valid_loader, 0):
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          self.val_losses.append(-self.loss_nored(scores, target_i))
          floss += -1 * self.loss(scores, target_i)

        self.val_losses = torch.cat(self.val_losses,0)

        N = self.curr_val_N
        grads_vec = [0 for _ in range(N)]
        for item in range(N):
            grads_vec[item] = torch.autograd.grad(self.val_losses[item],self.last_params , retain_graph=True) #self.model.parameters()

        self.grads_curr_subset = grads_vec
        #torch.autograd.grad(floss/(i+1), self.last_params)#self.model.parameters())

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
        #remainSet = set(range(self.curr_N))
        remainList = list(range(self.curr_N))
        #t_ng_start = time.time()    # naive greedy start time

        flat_grads_val = self.flatten_params(self.grads_curr_subset).to(self.device) ## Total parameter size
        #flat_grads_val = flat_grads_val.to(self.device)
        num_params = flat_grads_val.shape[0]

        flat_grads_mat = torch.zeros((self.N_trn, num_params))
        for i in remainList:
          flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])
        flat_grads_mat = (-1.0 * self.eta * flat_grads_mat).to(self.device)

        bestGain = -np.inf # value for current iteration (validation loss)

        #idxs_remain = list(remainSet)
        all_gains = torch.matmul(flat_grads_mat[remainList], flat_grads_val)
        
        # print(all_gains.shape)
        tmpid = torch.argmax(all_gains).item()
        bestId = remainList[tmpid]
        #bestGain = all_gains[tmpid]

        greedySet.add(bestId)
        remainList.remove(bestId)  
        numSelected = 1

        grads_currX = self._update_gradients_subset(grads_currX, bestId, theta_init)    
        flat_val_grads_mat = torch.zeros((self.curr_val_N, num_params))
        for i in range(self.curr_val_N):
          flat_val_grads_mat[i] = self.flatten_params(self.grads_curr_subset[i])
        flat_val_grads_mat = (flat_val_grads_mat).to(self.device)
        

        while(numSelected < budget):
            # Try Using a List comprehension here!
            #bestGain = -np.inf # value for current iteration (validation loss)

            #idxs_remain = list(remainSet)
            tay_approx = self.val_losses+torch.matmul(flat_grads_mat[remainList],torch.transpose(flat_val_grads_mat,0,1))
            all_gains = (torch.max(self.val_losses,tay_approx) -self.val_losses).sum(axis=0)
                      
            print(all_gains.shape)
            #print(self.curr_N)
            tmpid = torch.argmax(all_gains).item()
            #print(all_gains.item(),end=",")
            #if bestGain < all_gains[tmpid].item() :
            bestId = remainList[tmpid]
            bestGain = all_gains[tmpid].item()
                
            print(bestGain,end=",")
            greedySet.add(bestId)
            remainList.remove(bestId)            
            grads_currX = self._update_gradients_subset(grads_currX, bestId,theta_init)    

            flat_val_grads_mat = torch.zeros((self.curr_val_N, num_params))
            for i in range(self.curr_val_N):
              flat_val_grads_mat[i] = self.flatten_params(self.grads_curr_subset[i])
            flat_val_grads_mat = (flat_val_grads_mat).to(self.device)

            #flat_grads_val = self.flatten_params(self.grads_curr_subset).to(self.device)                 
            
            numSelected += 1
        #print("Naive greedy total time with taylor:", time.time()-t_ng_start)

        return list(greedySet)#, grads_currX

class SetFunctionTaylorDeep_Super(object):
    
    def __init__(self, x_trn, y_trn, trn_batch, x_val, y_val,valid, model, loss_criterion, loss_nored, eta,device,N_trn):

        self.x_trn = x_trn
        self.y_trn = y_trn
        self.trn_batch = trn_batch

        self.valid = valid
        self.x_val = x_val
        self.y_val = y_val

        self.model = model
        self.loss = loss_criterion # For validation loss
        self.loss_nored = loss_nored # Make sure it has reduction='none' instead of default
        self.eta = eta # step size for the one step gradient update
        self.N_trn = N_trn
        self.grads_per_elem = None
        self.first_element = True
        self.grads_val_curr = None
        self.device = device

        #print(len(self.valid_loader))
        #print(len(next(iter(self.valid_loader))[0]))
        #print(next(iter(self.valid_loader))[0][0])

    def class_wise(self,bud,theta_init):

      classes = torch.unique(self.y_trn)      

      self.N = self.y_trn.shape[0]
      greedyList =[]

      for cl in classes:

        idx = (self.y_trn == cl).nonzero().flatten()
        idx.tolist()

        curr_x_trn = self.x_trn[idx]
        curr_y_trn = self.y_trn[idx]
        self.curr_N = curr_y_trn.shape[0]
        #self.curr_bud = math.ceil(bud*self.curr_N / self.N)

        self.train_loader =[]
        for item in range(math.ceil(self.curr_N /self.trn_batch)):
          inputs = curr_x_trn[item*self.trn_batch:(item+1)*self.trn_batch]
          target  = curr_y_trn[item*self.trn_batch:(item+1)*self.trn_batch]
          self.train_loader.append((inputs,target))

        if self.valid:
          ind = (self.y_val == cl).nonzero().flatten()
          ind.tolist()

          curr_x_val = self.x_val[ind]
          curr_y_val = self.y_val[ind]

          self.valid_loader =[]
          for item in range(math.ceil(len(curr_y_val) /self.trn_batch)): #curr_y_val.shape[0]
            inputs = curr_x_val[item*self.trn_batch:(item+1)*self.trn_batch]
            target  = curr_y_val[item*self.trn_batch:(item+1)*self.trn_batch]
            self.valid_loader.append((inputs,target))

        else:
          self.valid_loader = self.train_loader
        
        subset = self.naive_greedy_max(math.ceil(bud*self.curr_N / self.N), theta_init)

        for j in range(len(subset)):
          greedyList.append(idx[subset[j]])

      greedyList.sort()
      return greedyList


    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        losses =[]
        for i, data_i in  enumerate(self.train_loader, 0):
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          losses.append(self.loss_nored(scores, target_i))

        losses = torch.cat(losses,0)

        N = self.curr_N
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
            grads_vec[item] = torch.autograd.grad(losses[item],self.last_params , retain_graph=True) #self.model.parameters()

        self.grads_per_elem = grads_vec


    def _compute_init_valloss_grads(self, theta_init):
        # Now compute the Validation loss initial gradient
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        floss = 0
        for i, data_i in  enumerate(self.valid_loader, 0):
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          floss += -1 * self.loss(scores, target_i)
        
        self.grads_curr_subset  = torch.autograd.grad(floss/(i+1), self.last_params)#self.model.parameters())
        self.first_element = False    
        

    def _update_gradients_subset(self, grads_X, element, theta_init):
        
        if not grads_X:
            grads_X = list(self.grads_per_elem[element])
        else:
            grads_e = self.grads_per_elem[element]
            for i, _ in enumerate(self.last_params):#self.model.parameters()):
                grads_X[i] += grads_e[i]

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.last_params):#self.model.parameters()):                                
                param.data.sub_(self.eta * grads_X[i])  
        # t = time.time()
        floss = 0
        for i, data_i in  enumerate(self.valid_loader, 0):
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          floss += -1 * self.loss(scores, target_i)

        self.grads_curr_subset = torch.autograd.grad(floss/(i+1), self.last_params)#self.model.parameters())
        
        '''losses =[]
        for i, data_i in  enumerate(self.train_loader, 0):
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          losses.append(self.loss_nored(scores, target_i))

        losses = torch.cat(losses,0)

        N = self.curr_N
        grads_vec = [0 for _ in range(N)]   # zero is just a placeholder

        for item in remain:
            grads_vec[item] = torch.autograd.grad(losses[item],self.last_params , retain_graph=True)#self.model.parameters()

        self.grads_per_elem = grads_vec''' 

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
        #remainSet = set(range(self.curr_N))
        remainList = list(range(self.curr_N))
        t_ng_start = time.time()    # naive greedy start time
        flat_grads_val = self.flatten_params(self.grads_curr_subset).to(self.device) ## Total parameter size
        #flat_grads_val = flat_grads_val.to(self.device)
        num_params = flat_grads_val.shape[0]

        flat_grads_mat = torch.zeros((self.N_trn, num_params))
        for i in remainList:
          flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])
        flat_grads_mat = (-1.0 * self.eta * flat_grads_mat).to(self.device)

        while(numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf # value for current iteration (validation loss)
            bestId = -1 # element to pick
            t_one_elem = time.time()

            #idxs_remain = list(remainSet)
            all_gains = torch.matmul(flat_grads_mat[remainList], flat_grads_val)
            
            # print(all_gains.shape)
            tmpid = torch.argmax(all_gains).item()
            bestId = remainList[tmpid]
            bestGain = all_gains[tmpid]

            greedySet.add(bestId)
            remainList.remove(bestId)            
            grads_currX = self._update_gradients_subset(grads_currX, bestId, theta_init)    
            flat_grads_val = self.flatten_params(self.grads_curr_subset).to(self.device)                 
            
            numSelected += 1
        #print("Naive greedy total time with taylor:", time.time()-t_ng_start)

        return list(greedySet)#, grads_currX



## One Step Set Functions on Validation Loss using just the last layer
class SetFunctionTaylorDeep(object):
    
    def __init__(self, train_loader, valid_loader, valid, model, loss_criterion, loss_nored, eta,device,N_trn):

        self.train_loader = train_loader
        
        if valid:
          self.valid_loader = valid_loader
        else:
          self.valid_loader = train_loader

        self.model = model
        self.loss = loss_criterion # For validation loss
        self.loss_nored = loss_nored # Make sure it has reduction='none' instead of default
        self.eta = eta # step size for the one step gradient update
        self.N_trn = N_trn
        self.grads_per_elem = None
        self.first_element = True
        self.grads_val_curr = None
        self.device = device

        #print(len(self.valid_loader))
        #print(len(next(iter(self.valid_loader))[0]))
        #print(next(iter(self.valid_loader))[0][0])


    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        losses =[]
        for i, data_i in  enumerate(self.train_loader, 0):
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          losses.append(self.loss_nored(scores, target_i))

        losses = torch.cat(losses,0)

        N = self.N_trn
        grads_vec = [0 for _ in range(N)]   # zero is just a placeholder

        '''for name,_ in self.model.named_children():
            self.last_layer_name = name

        self.last_params =[]
        for name, param in self.model.named_parameters():
            if self.last_layer_name not in name :
                param.requires_grad = False
            else:
                self.last_params.append(param)'''

        for item in range(N):
            grads_vec[item] = torch.autograd.grad(losses[item], self.model.parameters(), retain_graph=True) #self.last_params

        self.grads_per_elem = grads_vec


    def _compute_init_valloss_grads(self, theta_init):
        # Now compute the Validation loss initial gradient
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        floss = 0
        for i, data_i in  enumerate(self.valid_loader, 0):
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          floss += -1 * self.loss(scores, target_i)
        
        self.grads_curr_subset  = torch.autograd.grad(floss/(i+1), self.model.parameters())#self.last_params)
        self.first_element = False    
        

    def _update_gradients_subset(self, grads_X, element,theta_init):
        
        if not grads_X:
            grads_X = list(self.grads_per_elem[element])
        else:
            grads_e = self.grads_per_elem[element]
            for i, _ in enumerate(self.model.parameters()):#self.last_params):
                grads_X[i] += grads_e[i]

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()): #self.last_params):                               
                param.data.sub_(self.eta * grads_X[i])  
        # t = time.time()
        floss = 0
        for i, data_i in  enumerate(self.valid_loader, 0):
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          floss += -1 * self.loss(scores, target_i)

        self.grads_curr_subset = torch.autograd.grad(floss/(i+1), self.model.parameters())#self.last_params)
        return grads_X
        # e = time.time() - t
        # print("grad update time:", e)
    
    def flatten_params(self, param_list):
        l = [torch.flatten(p) for p in param_list]
        flat = torch.cat(l)
        return flat


    def naive_greedy_max(self, budget, theta_init,remainList=None):
        self._compute_per_element_grads(theta_init)
        self._compute_init_valloss_grads(theta_init)
        #print("Computed train set gradients")
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        #remainSet = set(range(self.N_trn))
        if remainList == None:
          remainList = list(range(self.N_trn))
        
        t_ng_start = time.time()    # naive greedy start time
        flat_grads_val = self.flatten_params(self.grads_curr_subset).to(self.device) ## Total parameter size
        #flat_grads_val = flat_grads_val.to(self.device)
        num_params = flat_grads_val.shape[0]

        flat_grads_mat = torch.zeros((self.N_trn, num_params))
        for i in range(self.N_trn):
            flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])
        flat_grads_mat = (-1.0 * self.eta * flat_grads_mat).to(self.device)
        
        while(numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf # value for current iteration (validation loss)
            bestId = -1 # element to pick
            t_one_elem = time.time()

            #idxs_remain = list(remainSet)
            all_gains = torch.matmul(flat_grads_mat[remainList], flat_grads_val)
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
        #if numSelected % 500 == 0:
        #      Printing bestGain and Selection time for 1 element.
        #      print("numSelected:", numSelected, "Time for 1:", time.time()-t_one_elem, "bestGain:", bestGain)
        return list(greedySet)#, grads_currX


class SetFunctionTaylor(object):
    
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


    def naive_greedy_max(self, budget, theta_init,remainList=None,previous=None,random=False):
        self._compute_per_element_grads(theta_init)
        self._compute_init_valloss_grads(theta_init)
        #print("Computed train set gradients")
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        #remainSet = set(range(self.N_trn))

        if remainList == None:
          remainList = list(range(self.N_trn))

        #print(len(remainList))
        
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
            #idxs_remain = list(remainSet)
            if previous == None:
              all_gains = torch.matmul(-1.0 * self.eta * flat_grads_mat[remainList], flat_grads_val)
            else:
              all_gains = torch.matmul(-1.0 * self.eta * flat_grads_mat[remainList], flat_grads_val) - previous[remainList]
            
            if random :
              all_gains = all_gains - self.eta*torch.randn(len(remainList), device= self.device)

            # print(all_gains.shape)
            tmpid = int(torch.argmax(all_gains))
            bestId = remainList[tmpid]
            bestGain = all_gains[tmpid]
            greedySet.add(bestId)
            remainList.remove(bestId)            
            grads_currX = self._update_gradients_subset(grads_currX, bestId, theta_init)                           
            # print("numSelected:", numSelected, "Time for 1:", time.time()-t_one_elem, "bestGain:", bestGain)
            '''if numSelected % 500 == 0:
                # Printing bestGain and Selection time for 1 element.
               print("numSelected:", numSelected, "Time for 1:", time.time()-t_one_elem, "bestGain:", bestGain)'''
            numSelected += 1
        print("Naive greedy total time with taylor:", time.time()-t_ng_start)

        # Do an update to model parameters based on grads from greedy subset
        # self.model.load_state_dict(theta_init)
        # with torch.no_grad(): # perform one-step update
        #     for i, param in enumerate(self.model.parameters()):
        #         param.data.sub_(self.eta * grads_currX[i])
        #     scores = self.model(self.x_val)
        #     loss = self.loss(scores, self.y_val)
        #     print("final val loss:", loss)
        return list(greedySet)
        # return greedySet, grads_currX



class SetFunctionCRAIG(object):

    def __init__(self, device ,train_full_loader,if_convex):#, valid_loader): 
        
        self.train_loader = train_full_loader      
        self.if_convex = if_convex
        self.device = device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def distance(self,x, y, exp = 2):

      n = x.size(0)
      m = y.size(0)
      d = x.size(1)

      x = x.unsqueeze(1).expand(n, m, d)
      y = y.unsqueeze(0).expand(n, m, d)

      dist = torch.pow(x - y, exp).sum(2) 
      return dist #torch.sqrt(dist)

      #dist = torch.abs(x-y).sum(2)
      #return dist

    def compute_score(self, model):

      self.N = 0
      g_is =[]

      with torch.no_grad():

        for i, data_i in  enumerate(self.train_loader, 0):
          inputs_i, target_i = data_i
          inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          #print(i,end=",")
          self.N +=  inputs_i.size(0)

          if not self.if_convex:
            scores_i = model(inputs_i)
            y_i = torch.zeros(target_i.size(0),scores_i.size(1)).to(self.device)
            y_i[range(y_i.shape[0]), target_i]=1

            g_is.append(scores_i - y_i)
          else:
            g_is.append(inputs_i)

        #temp = torch.stack(g_is[:-1])
        #print(g_is.shape)

        self.dist_mat = torch.zeros([self.N, self.N],dtype=torch.float32)

        first_i = True

        for i, g_i in  enumerate(g_is, 0):

          #print(i,end=",")
          if first_i:
            size_b = g_i.size(0)
            first_i = False

          for j, g_j in  enumerate(g_is, 0):

            self.dist_mat[i*size_b: i*size_b + g_i.size(0) ,j*size_b: j*size_b + g_j.size(0)] = self.distance(g_i, g_j)
        
      dist = self.dist_mat.sum(1)
      bestId = torch.argmin(dist).item()

      self.min_dist = self.dist_mat[bestId]

      return bestId

    def compute_gamma(self,idxs):

      gamma = [0 for i in range(len(idxs))]

      best = self.dist_mat[idxs]
      #print(best[0])
      rep = torch.argmin(best,axis = 0)

      for i in rep:
        gamma[i] += 1

      return gamma
    

    def naive_greedy_max(self, budget, model):

      id_first = self.compute_score(model)
      #print("model updated")

      numSelected = 1
      greedyList = [id_first]  
      not_selected = [i for i in range(self.N)]
      #print(not_selected)
      not_selected.remove(id_first)

      while(numSelected < budget):        
          bestGain = -np.inf
          
          for i in not_selected:
            gain = (self.min_dist - torch.min(self.min_dist, self.dist_mat[i])).sum()#L([0]+greedyList+[i]) - L([0]+greedyList)  #s_0 = self.x_trn[0]
            if bestGain < gain.item():
              bestGain = gain.item()
              bestId = i

          greedyList.append(bestId)
          not_selected.remove(bestId)
          numSelected += 1

          self.min_dist = torch.min(self.min_dist,self.dist_mat[bestId])

      gamma = self.compute_gamma(greedyList)
      return greedyList, gamma

    def lazy_greedy_max(self, budget, model):

      id_first = self.compute_score(model)
      #print("model updated")

      self.gains = PriorityQueue()
      for i in range(self.N):
        
        if i == id_first :
          continue
        curr_gain = (self.min_dist - torch.min(self.min_dist,self.dist_mat[i])).sum()
        self.gains.put((-curr_gain.item(),i))

      numSelected = 2
      second = self.gains.get()
      greedyList = [id_first,second[1]]
      self.min_dist = torch.min(self.min_dist,self.dist_mat[second[1]])

      while(numSelected < budget):

          #print(len(greedyList)) 
          if self.gains.empty():
            break

          elif self.gains.qsize() == 1:
            bestId = self.gains.get()[1]

          else:
   
            bestGain = -np.inf
            bestId = None
            
            while True:

              first = self.gains.get()
              if bestId == first[1]: 
                break

              curr_gain = (self.min_dist - torch.min(self.min_dist,self.dist_mat[first[1]])).sum()
              self.gains.put((-curr_gain.item(), first[1]))

              if curr_gain.item() >= bestGain:
                bestGain = curr_gain.item()
                bestId = first[1]

          greedyList.append(bestId)
          numSelected += 1

          self.min_dist = torch.min(self.min_dist,self.dist_mat[bestId])

      gamma = self.compute_gamma(greedyList)
      return greedyList, gamma



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
    # Note that it modifies the input e vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        for i, _ in enumerate(self.model.parameters()):
            grads_X[i] += grads_e[i]


    def naive_greedy_max(self, budget, theta_init):
        self._compute_per_element_grads(theta_init)
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


class SetFunctionCRAIG_Super(object):

    def __init__(self, device ,X_trn, Y_trn,if_convex):#, valid_loader): 
        
        self.x_trn = X_trn
        self.y_trn = Y_trn      
        self.if_convex = if_convex
        self.device = device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def distance(self,x, y, exp = 2):

      n = x.size(0)
      m = y.size(0)
      d = x.size(1)

      x = x.unsqueeze(1).expand(n, m, d)
      y = y.unsqueeze(0).expand(n, m, d)

      dist = torch.pow(x - y, exp).sum(2) 
      return dist #torch.sqrt(dist)

      #dist = torch.abs(x-y).sum(2)
      #return dist

    def class_wise(self,bud,model):

      torch.cuda.empty_cache()

      classes = torch.unique(self.y_trn)      

      self.N = self.y_trn.shape[0]
      greedyList =[]
      full_gamma= []

      for i in classes:

        idx = (self.y_trn == i).nonzero().flatten()
        idx.tolist()

        self.curr_x_trn = self.x_trn[idx]
        self.curr_y_trn = self.y_trn[idx]
        self.curr_N = self.curr_y_trn.shape[0]
        #self.curr_bud = math.ceil(bud*self.curr_N / self.N)

        id_first = self.compute_score(model)
        subset, gamma = self.lazy_greedy_max(math.ceil(bud*self.curr_N / self.N), id_first)

        for j in range(len(subset)):
          greedyList.append(idx[subset[j]])
          full_gamma.append(gamma[j])

      return greedyList, full_gamma


    def compute_score(self, model):

      with torch.no_grad():

        self.dist_mat = torch.zeros([self.curr_N, self.curr_N],dtype=torch.float32)

        train_batch_size = 1200
        train_loader = []
        for item in range(math.ceil(self.curr_N/train_batch_size)):
          inputs = self.curr_x_trn[item*train_batch_size:(item+1)*train_batch_size]
          target  = self.curr_y_trn[item*train_batch_size:(item+1)*train_batch_size]
          train_loader.append((inputs,target))

        first_i = True
        g_is =[]

        for i, data_i in  enumerate(train_loader, 0): #iter(train_loader).next()

          inputs_i, target_i = data_i
          inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          
          if not self.if_convex:
            scores_i = model(inputs_i)
            y_i = torch.zeros(target_i.size(0),scores_i.size(1)).to(self.device)
            y_i[range(y_i.shape[0]), target_i]=1

            g_is.append(scores_i - y_i)
          else:
            g_is.append(inputs_i)


        first_i = True
        for i, g_i in  enumerate(g_is, 0):

          #print(i,end=",")
          if first_i:
            size_b = g_i.size(0)
            first_i = False

          for j, g_j in  enumerate(g_is, 0):
              self.dist_mat[i*size_b: i*size_b + g_i.size(0) ,j*size_b: j*size_b + g_j.size(0)] = self.distance(g_i, g_j)

      dist = self.dist_mat.sum(1)
      bestId = torch.argmin(dist).item()

      self.dist_mat = self.dist_mat.to(self.device)

      self.min_dist = self.dist_mat[bestId].to(self.device)

      return bestId

    def compute_gamma(self,idxs):

      gamma = [0 for i in range(len(idxs))]

      best = self.dist_mat[idxs]
      #print(best[0])
      rep = torch.argmin(best,axis = 0)

      for i in rep:
        gamma[i] += 1

      return gamma
    
    def lazy_greedy_max(self, budget, id_first):


      self.gains = PriorityQueue()
      for i in range(self.curr_N):
        
        if i == id_first :
          continue
        curr_gain = (self.min_dist - torch.min(self.min_dist,self.dist_mat[i])).sum()
        self.gains.put((-curr_gain.item(),i))

      numSelected = 2
      second = self.gains.get()
      greedyList = [id_first,second[1]]
      self.min_dist = torch.min(self.min_dist,self.dist_mat[second[1]])

      while(numSelected < budget):

          #print(len(greedyList)) 
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

              curr_gain = (self.min_dist - torch.min(self.min_dist,self.dist_mat[first[1]])).sum()
              self.gains.put((-curr_gain.item(), first[1]))


              if curr_gain.item() >= bestGain:
                  
                bestGain = curr_gain.item()
                bestId = first[1]

          greedyList.append(bestId)
          numSelected += 1

          self.min_dist = torch.min(self.min_dist,self.dist_mat[bestId])

      gamma = self.compute_gamma(greedyList)
      return greedyList, gamma


'''
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
'''


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





class SetFunctionTaylorMeta(object):
    
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

            # for i in remainSet:
            #     grads_i = self.flatten_params(self.grads_per_elem[i])
            #     val_i = self.eval(flat_grads_val, grads_i, theta_init)
            #     if val_i > bestGain:
            #         bestGain = val_i
            #         bestId = i

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

        # Do an update to model parameters based on grads from greedy subset
        # self.model.load_state_dict(theta_init)
        # with torch.no_grad(): # perform one-step update
        #     for i, param in enumerate(self.model.parameters()):
        #         param.data.sub_(self.eta * grads_currX[i])
        #     scores = self.model(self.x_val)
        #     loss = self.loss(scores, self.y_val)
        #     print("final val loss:", loss)


        return list(greedySet)
        # return greedySet, grads_currX
