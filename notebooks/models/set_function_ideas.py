import numpy as np
import torch
import torch.nn as nn
import copy
import time

import math
from queue import PriorityQueue
import random

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

          if len(ind) != 0: 
            curr_x_val = self.x_val[ind]
            curr_y_val = self.y_val[ind]
            self.curr_val_N = len(curr_y_val)
          else:
            curr_x_val = self.x_val
            curr_y_val = self.y_val
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

      #greedyList.sort()
      random.shuffle(greedyList)
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

        '''for name,_ in self.model.named_children():
            self.last_layer_name = name

        self.last_params =[]
        for name, param in self.model.named_parameters():
            if self.last_layer_name not in name :
                param.requires_grad = False
            else:
                self.last_params.append(param)'''

        for item in range(N):
        	for param in self.model.parameters():
        		param.grad.zero_()

        	grads_vec[item] = torch.autograd.grad(losses[item],self.model.parameters(), retain_graph=True) #self.last_params

        self.grads_per_elem = grads_vec


    def _compute_init_valloss_grads(self, theta_init):
        # Now compute the Validation loss initial gradient
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        #floss = 0
        self.val_losses =[]
        for i, data_i in  enumerate(self.valid_loader, 0):

          self.model.zero_grad() 
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          self.val_losses.append(self.loss_nored(scores, target_i))
          #floss += -1 * self.loss(scores, target_i)

        #print(self.val_losses[0].grad_fn)

        self.val_losses = torch.cat(self.val_losses,0)
        N = self.curr_val_N
        grads_vec = [0 for _ in range(N)]
        for item in range(N):
        	for param in self.model.parameters():
        		param.grad.zero_()
        	grads_vec[item] = torch.autograd.grad(self.val_losses[item],self.model.parameters() , retain_graph=True) #self.last_params

        self.grads_curr_subset = grads_vec
        #torch.autograd.grad(floss/(i+1), self.last_params)#self.model.parameters())
        self.first_element = False    
        

    def _update_gradients_subset(self, grads_X, element, theta_init):
        
        if not grads_X:
            grads_X = list(self.grads_per_elem[element])
        else:
            grads_e = self.grads_per_elem[element]
            for i, _ in enumerate(self.model.parameters()): #self.last_params
                grads_X[i] += grads_e[i]

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()): #self.last_params                             
                param.data.sub_(self.eta * grads_X[i])  
        # t = time.time()
        #floss = 0
        self.val_losses = []
        for i, data_i in  enumerate(self.valid_loader, 0):
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          self.val_losses.append(self.loss_nored(scores, target_i))
          #floss += -1 * self.loss(scores, target_i)

        self.val_losses = torch.cat(self.val_losses,0)

        N = self.curr_val_N
        grads_vec = [0 for _ in range(N)]
        for item in range(N):
        	for param in self.model.parameters():
        		param.grad.zero_()
        	grads_vec[item] = torch.autograd.grad(self.val_losses[item],self.model.parameters(), retain_graph=True) #self.last_params

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

        #flat_grads_val = self.flatten_params(self.grads_curr_subset).to(self.device) ## Total parameter size
        #flat_grads_val = flat_grads_val.to(self.device)
        num_params = self.flatten_params(self.grads_curr_subset[0]).shape[0]

        flat_grads_mat = torch.zeros((self.curr_N, num_params))
        for i in remainList:
          flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])
        flat_grads_mat = (-1.0 * self.eta * flat_grads_mat).to(self.device)
   
        flat_val_grads_mat = torch.zeros((self.curr_val_N, num_params))
        for i in range(self.curr_val_N):
          flat_val_grads_mat[i] = self.flatten_params(self.grads_curr_subset[i])
        flat_val_grads_mat = (flat_val_grads_mat).to(self.device)
        
        #path = '/home/durga/Documents/Phd_Work/Code/Main/datk-dss/data-selection/results/temp/satimage/detilas.txt'
        #logfile = open(path,'a')
        while(numSelected < budget):
            # Try Using a List comprehension here!
            #bestGain = -np.inf # value for current iteration (validation loss)

            #idxs_remain = list(remainSet)
            #for i in self.val_losses:
            # 	if i < 0:
            #		print(i)
            
            tay_approx = self.val_losses + torch.matmul(flat_grads_mat[remainList],torch.transpose(flat_val_grads_mat,0,1))
            #

            all_gains = (self.val_losses - torch.min(self.val_losses,tay_approx)).sum(axis=1)

            #print(tay_approx[0][1:10])
            #print(self.val_losses[1:10])
            #print(torch.min(self.val_losses,tay_approx)[0][1:10])
            #print((self.val_losses -torch.min(self.val_losses,tay_approx)).shape)

            #print(all_gains.shape)
            #print(self.curr_N)
            tmpid = torch.argmax(all_gains).item()
            #print(all_gains.item(),end=",")
            #if bestGain < all_gains[tmpid].item() :
            bestId = remainList[tmpid]
            bestGain = all_gains[tmpid].item()
                
            print(tay_approx[bestId][1:10])
            #print(bestId,bestGain)#,end=",",file=logfile)
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
        #print("",file=logfile)
        return list(greedySet)#, grads_currX

class SetFunctionTaylorDeep_SuperDistance(object):
    
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

          if len(ind) != 0: 
            curr_x_val = self.x_val[ind]
            curr_y_val = self.y_val[ind]
            self.curr_val_N = len(curr_y_val)
          else:
            curr_x_val = self.x_val
            curr_y_val = self.y_val
            self.curr_val_N = len(curr_y_val)        

          self.valid_loader =[]
          for item in range(math.ceil(len(curr_y_val) /self.trn_batch)): #curr_y_val.shape[0]
            inputs = curr_x_val[item*self.trn_batch:(item+1)*self.trn_batch]
            target  = curr_y_val[item*self.trn_batch:(item+1)*self.trn_batch]
            self.valid_loader.append((inputs,target))

        else:
          self.curr_val_N = self.curr_N
          self.valid_loader = self.train_loader
        
        id_first = self.compute_score(theta_init)
        #print(cl)
        subset = self.lazy_greedy_max(math.ceil(bud*self.curr_N / self.N), id_first)

        for j in range(len(subset)):
          greedyList.append(idx[subset[j]])

      #greedyList.sort()
      random.shuffle(greedyList)
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
            grads_vec[item] = torch.autograd.grad(losses[item],self.model.parameters(), retain_graph=True) #self.model.parameters()

        self.grads_per_elem = grads_vec


    def _compute_init_valloss_grads(self, theta_init):
        # Now compute the Validation loss initial gradient
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        #floss = 0
        self.val_losses =[]
        for i, data_i in  enumerate(self.valid_loader, 0):
          inputs_i, target_i = data_i
          #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          self.val_losses.append(-1.0*self.loss_nored(scores, target_i))
          #floss += -1 * self.loss(scores, target_i)

        self.val_losses = torch.cat(self.val_losses,0)
        N = self.curr_val_N
        grads_vec = [0 for _ in range(N)]
        for item in range(N):
            grads_vec[item] = torch.autograd.grad(self.val_losses[item],self.model.parameters(), retain_graph=True) #self.last_params

        self.grads_curr_subset = grads_vec
        #torch.autograd.grad(floss/(i+1), self.last_params)#self.model.parameters())
        self.first_element = False    

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

    def compute_score(self, theta_init):

    	self._compute_per_element_grads(theta_init)
    	self._compute_init_valloss_grads(theta_init)

    	num_params = self.flatten_params(self.grads_curr_subset[0]).shape[0]
    	#print(num_params)
    	flat_grads_mat = torch.zeros((self.curr_N, num_params))

    	for i in range(self.curr_N):
    		flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])

    	flat_grads_mat = (flat_grads_mat).to(self.device)

    	dist_trn = torch.norm(flat_grads_mat,dim=1)
    	flat_grads_mat = flat_grads_mat/dist_trn.reshape((self.curr_N,1))

    	flat_val_grads_mat = torch.zeros((self.curr_val_N, num_params))
    	for i in range(self.curr_val_N):
    		flat_val_grads_mat[i] = self.flatten_params(self.grads_curr_subset[i])

    	flat_val_grads_mat = (flat_val_grads_mat).to(self.device)

    	dist_val = torch.norm(flat_val_grads_mat,dim=1)
    	flat_val_grads_mat = flat_val_grads_mat/dist_val.reshape((self.curr_val_N,1))
    	#print(flat_val_grads_mat.shape)
    	#print(dist_val.shape)

    	b_size = 128 #self.trn_batch
    	self.dist_mat = torch.zeros([self.curr_N, self.curr_val_N],dtype=torch.float32)

    	for i in range(math.ceil(self.curr_N/b_size)):
    		g_i = flat_grads_mat[i*b_size: (i+1)*b_size]

    		for j in range(math.ceil(self.curr_val_N/b_size)):
    			g_j = flat_val_grads_mat[j*b_size: (j+1)*b_size]
    			self.dist_mat[i*b_size: (i+1)*b_size,j*b_size: (j+1)*b_size] = self.distance(g_i, g_j)

    	dist = self.dist_mat.sum(1)
    	bestId = torch.argmin(dist).item()
    	self.dist_mat = self.dist_mat.to(self.device)
    	self.min_dist = self.dist_mat[bestId].to(self.device)

    	return bestId

    def flatten_params(self, param_list):
        l = [torch.flatten(p) for p in param_list]
        flat = torch.cat(l)
        return flat

   
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

      return greedyList
    
    
class SetFunctionTaylorDeep_Super_ReLoss(object):
    
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


    def class_wise(self,bud,theta_init):

      classes = torch.unique(self.y_trn)     
      #self.no_of_classes = classes.shape[0] 

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

          if len(ind) != 0: 
            curr_x_val = self.x_val[ind]
            curr_y_val = self.y_val[ind]
            self.curr_val_N = curr_y_val.shape[0]
          else:
            curr_x_val = self.x_val
            curr_y_val = self.y_val
            self.curr_val_N = curr_y_val.shape[0]

          self.valid_loader =[]
          for item in range(math.ceil(len(curr_y_val) /self.trn_batch)): #curr_y_val.shape[0]
            inputs = curr_x_val[item*self.trn_batch:(item+1)*self.trn_batch]
            target  = curr_y_val[item*self.trn_batch:(item+1)*self.trn_batch]
            self.valid_loader.append((inputs,target))

        else:
          self.valid_loader = self.train_loader
          self.curr_val_N = self.curr_N
        
        subset = self.naive_greedy_max(math.ceil(bud*self.curr_N / self.N), theta_init)

        for j in range(len(subset)):
          greedyList.append(idx[subset[j]])

      random.shuffle(greedyList)
      return greedyList


    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
          
        losses =[]
        for i, data_i in  enumerate(self.train_loader, 0):
          #inputs_i, target_i = data_i
          inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
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

        #make_dot(losses[0]).render("loss_graph", format="pdf")

        for item in range(N):
            grads_vec[item] = torch.autograd.grad(losses[item],self.last_params , retain_graph=True) #self.model.parameters()

        self.grads_per_elem = grads_vec


    def _compute_init_valloss_grads(self, theta_init):
        # Now compute the Validation loss initial gradient
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        #floss = 0
        scores_list = []
        target_list = []
        for i, data_i in  enumerate(self.valid_loader, 0):
          #inputs_i, target_i = data_i
          inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          scores_list.append(scores)
          target_list.append(target_i)
          #floss += -1 * self.loss(scores, target_i)

        scores_list = torch.cat(scores_list,0)
        self.logits = scores_list
        self.targets = torch.cat(target_list,0)

        N = self.curr_val_N 
        self.no_cl =  scores_list.shape[1]
        grads_vec = [[0 for _ in range(self.no_cl)] for _ in range(N)]   # zero is just a placeholder
        
        for item in range(N):
          for val in range(self.no_cl):
            grads_vec[item][val] = torch.autograd.grad(scores_list[item][val], self.last_params, retain_graph=True) #self.model.parameters()
        
        self.grads_curr_subset  = grads_vec
        #torch.autograd.grad(floss/(i+1),self.model.parameters())#self.last_params
        

    def _update_gradients_subset(self, grads_X, element, theta_init):
        
        if not grads_X:
            grads_X = list(self.grads_per_elem[element])
        else:
            grads_e = self.grads_per_elem[element]
            for i, _ in enumerate(self.last_params): #self.model.parameters()):
                grads_X[i] += grads_e[i]

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.last_params): #self.model.parameters()):                         
                param.data.sub_(self.eta * grads_X[i])  
        # t = time.time()
        #floss = 0
        scores_list = []
        #target_list = []
        for i, data_i in  enumerate(self.valid_loader, 0):
          #inputs_i, target_i = data_i
          inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          scores_list.append(scores)
          #target_list.append(target_i)
          #floss += -1 * self.loss(scores, target_i)

        scores_list = torch.cat(scores_list,0)
        self.logits = scores_list
        #self.targets = torch.cat(target_list,0)

        N = self.curr_val_N 
        self.no_cl =  scores_list.shape[1]

        grads_vec = [[0 for _ in range(self.no_cl)] for _ in range(N)]   # zero is just a placeholder
        
        for item in range(N):
          for val in range(self.no_cl):
            grads_vec[item][val] = torch.autograd.grad(scores_list[item][val], self.last_params, retain_graph=True) #self.model.parameters()
        
        self.grads_curr_subset  = grads_vec

        #self.grads_curr_subset = torch.autograd.grad(floss/(i+1), self.model.parameters())#self.last_params

        return grads_X
        

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

        #print(torch.tensor(self.grads_curr_subset).shape)

        N = self.curr_val_N 
        num_params = self.flatten_params(self.grads_curr_subset[0][0]).shape[0]

        flat_grads_val = torch.zeros(N, self.no_cl, num_params)
        for item in range(N):
          for val in range(self.no_cl):
            flat_grads_val[item][val] = self.flatten_params(self.grads_curr_subset[item][val])  ## Total parameter size

        #print("Flattening time:", time.time()-t_ng_start)
        #t_ng_start = time.time()
        
        flat_grads_val = flat_grads_val.to(self.device)

        flat_grads_mat = torch.zeros((self.curr_N, num_params))

        for i in remainList:
          flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])
        flat_grads_mat = (-1.0 * self.eta * flat_grads_mat).to(self.device)

        #print("Flattening time trn:", time.time()-t_ng_start)
        #t_ng_start = time.time()

        
        while(numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf # value for current iteration (validation loss)
            bestId = -1 # element to pick
            #t_one_elem = time.time()

            #idxs_remain = list(remainSet)
            
            all_gains = torch.matmul(flat_grads_mat[remainList],torch.transpose(flat_grads_val,1,2))
            all_gains = torch.transpose(all_gains,0,1) + self.logits

            #print("Multiplying time :", time.time()-t_ng_start)
            #t_ng_start = time.time()

            losses = torch.zeros(len(remainList))
            for i in range(len(remainList)):
              losses[i] = -1 * self.loss(all_gains[i], self.targets)

            #print("Loss time :", time.time()-t_ng_start)
            #t_ng_start = time.time()
            
            # print(all_gains.shape)
            tmpid = torch.argmax(losses).item()
            bestId = remainList[tmpid]
            bestGain = all_gains[tmpid]

            greedySet.add(bestId)
            remainList.remove(bestId)            
            grads_currX = self._update_gradients_subset(grads_currX, bestId, theta_init)    
            for item in range(N):
              for val in range(self.no_cl):
                flat_grads_val[item][val] = self.flatten_params(self.grads_curr_subset[item][val])    
            flat_grads_val = flat_grads_val.to(self.device)

            print("Flattening time trn:", time.time()-t_ng_start)
            t_ng_start = time.time()          
            
            numSelected += 1
            #print(numSelected)
        #print("Naive greedy total time with taylor:", time.time()-t_ng_start)

        return list(greedySet)#, grads_currX

class SetFunctionTaylorDeep_Super_ReLoss_Mean(object):
    
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


    def class_wise(self,bud,theta_init):

      """loss_class = []

      if self.valid:

        classes = torch.unique(self.y_val)

        for cl in classes:

          idx = (self.y_val == cl).nonzero().flatten()
          idx.tolist()

          self.model.load_state_dict(theta_init)
          self.model.zero_grad()

          inputs = self.x_val[idx]
          target = self.y_val[idx]
          scores = self.model(inputs)
          #loss_class.append(self.loss(scores, target).item())
          
          _, predicted = scores.max(1)
          val_total = target.size(0)
          val_correct = predicted.eq(target).sum().item()
          loss_class.append(1-val_correct/val_total)
          #loss_class.append(val_total-val_correct)
           

        val = max(loss_class) + 1 #min(loss_class) - 1
        for i in torch.unique(self.y_trn):
          if i not in classes:
            print(i)
            loss_class.append(val)
            classes.append(i)

      else:

        classes = torch.unique(self.y_trn)

        for cl in classes:

          idx = (self.y_trn == cl).nonzero().flatten()
          idx.tolist()

          self.model.load_state_dict(theta_init)
          self.model.zero_grad()

          inputs = self.x_trn[idx]
          target = self.y_trn[idx]
          scores = self.model(inputs)
          #loss_class.append(self.loss(scores, target).item())      

          _, predicted = scores.max(1)
          val_total = target.size(0)
          val_correct = predicted.eq(target).sum().item()
          loss_class.append(1-val_correct/val_total)
          #loss_class.append(val_total-val_correct)
        
      print(loss_class)

      classes = [x for _,x in sorted(zip(loss_class,classes))]#,reverse=True)]  
      loss_class = sorted(loss_class)#,reverse=True)   

      #classes,count = torch.unique(self.y_trn,return_counts=True)
      #classes = [x for _,x in sorted(zip(count,classes))]
      #loss_class = [x for _,x in sorted(zip(count,loss_class))]#,reverse=True)]
      
      print(classes)
      #self.no_of_classes = classes.shape[0]

      denom = sum(loss_class)"""

      classes = torch.unique(self.y_trn)

      #self.grads_currX = [] 

      self.N = self.y_trn.shape[0]
      greedyList =[]

      self.no_cl = len(classes)

      diff = 0

      for cl in classes:

        idx = (self.y_trn == cl).nonzero().flatten()
        idx.tolist()

        self.curr_class = cl

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

          if len(ind) != 0: 
            curr_x_val = self.x_val[ind]
            curr_y_val = self.y_val[ind]
            self.curr_val_N = curr_y_val.shape[0]

            self.valid_loader =[]
            for item in range(math.ceil(len(curr_y_val) /self.trn_batch)): #curr_y_val.shape[0]
              inputs = curr_x_val[item*self.trn_batch:(item+1)*self.trn_batch]
              target  = curr_y_val[item*self.trn_batch:(item+1)*self.trn_batch]
              self.valid_loader.append((inputs,target))
          
          else:
            self.valid_loader = self.train_loader
            self.curr_val_N = self.curr_N       

        else:
          self.valid_loader = self.train_loader
          self.curr_val_N = self.curr_N

        #if denom != 0:
        #  budget = bud*0.9*self.curr_N/self.N + bud*0.1*loss_class[classes.index(cl)]/denom + diff
        #else:
        budget = bud*self.curr_N/self.N

        if budget < len(idx):
          subset = self.naive_greedy_max(math.ceil(budget), theta_init) #math.ceil(bud*loss_class[classes.index(cl)]/denom)
          #diff = 0
        else:
          subset = [i for i in range(len(idx))]
          #diff = budget - len(idx)
          #print(str(diff)+" "+str(cl))
        #print([idx[i] for i in subset[:15]])
        '''print(self.valid_loader[0][0][0])
        print(self.train_loader[0][0][0])'''

        for j in range(len(subset)):
          greedyList.append(idx[subset[j]])

      random.shuffle(greedyList)
      return greedyList


    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
          
        losses =[]
        for i, data_i in  enumerate(self.train_loader, 0):
          inputs_i, target_i = data_i
          inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          losses.append(self.loss_nored(scores, target_i))

        losses = torch.cat(losses,0)

        N = self.curr_N
        grads_vec = [0 for _ in range(N)]   # zero is just a placeholder

        #print(self.model.linear2.bias)

        '''for name, _ in self.model.named_children():
            self.last_layer_name = name

        self.last_params =[]
        for name, param in self.model.named_parameters():
            if self.last_layer_name not in name :
                param.requires_grad = False
            else:
                self.last_params.append(param)
                #print(param)'''


        for item in range(N):
            grads_vec[item] = torch.autograd.grad(losses[item], self.model.parameters(), retain_graph=True) #self.model.parameters(),

        #print(self.model.linear2.bias)

        self.grads_per_elem = grads_vec


    def _compute_init_valloss_grads(self, theta_init):
        # Now compute the Validation loss initial gradient
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        '''if self.grads_currX:
          with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()):                        
                param.data.sub_(self.eta * self.grads_currX[i])''' 

        #floss = 0
        self.total = 0
        self.sum_score = torch.zeros(self.no_cl)
        self.sum_score = self.sum_score.to(self.device)
        for i, data_i in  enumerate(self.valid_loader, 0):
          inputs_i, target_i = data_i
          inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          self.sum_score += torch.sum(scores, 0)
          self.target = target_i[0]
          self.total+=target_i.shape[0]


        #print(total)
        #print(self.sum_score)  
        self.sum_score = self.sum_score/self.total
        #print(self.sum_score)  

        self.num_params = self.flatten_params(self.grads_per_elem[0]).shape[0]

        grads_vec = [ 0 for _ in range(self.no_cl)]   # zero is just a placeholder

        #print(self.model.linear2.bias)
        
        for val in range(self.no_cl):
            grads_vec[val] = torch.autograd.grad(self.sum_score[val], self.model.parameters(), retain_graph=True,allow_unused=True) #model.parameters()
        
        #print(self.model.linear2.bias)
        self.grads_curr_subset  = grads_vec
        #torch.autograd.grad(floss/(i+1),self.model.parameters())#self.last_params
        

    def _update_gradients_subset(self, grads_X, element, theta_init):
        
        if not grads_X:
            grads_X = list(copy.deepcopy(self.grads_per_elem[element]))
        else:
            grads_e = self.grads_per_elem[element]
            for i, _ in enumerate(self.model.parameters()): 
                grads_X[i] += grads_e[i]

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()):                           
                param.data.sub_(self.eta * grads_X[i]) 
        

        self.sum_score = torch.zeros(self.no_cl)
        self.sum_score = self.sum_score.to(self.device)
        for i, data_i in  enumerate(self.valid_loader, 0):
          inputs_i, target_i = data_i
          inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          self.sum_score += torch.sum(scores, 0)

        self.sum_score = self.sum_score/self.total
        #print(self.sum_score)

        grads_vec = [0 for _ in range(self.no_cl)]   # zero is just a placeholder
        
        #print(self.model.linear2.bias)
        for val in range(self.no_cl):
            grads_vec[val] = torch.autograd.grad(self.sum_score[val], self.model.parameters(), retain_graph=True) #self.model.parameters()
        
        self.grads_curr_subset  = grads_vec

        #for name, param in self.model.named_parameters():
        #    print(param.grad.data)

        #self.grads_curr_subset = torch.autograd.grad(floss/(i+1), self.model.parameters())#self.last_params
        #print(self.model.linear2.bias)
        return grads_X
        

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

        #print(torch.tensor(self.grads_curr_subset).shape)

        #N = self.curr_val_N 

        flat_grads_val = torch.zeros(self.no_cl, self.num_params)
        for val in range(self.no_cl):
            flat_grads_val[val] = self.flatten_params(self.grads_curr_subset[val])  ## Total parameter size


        #print("Flattening time:", time.time()-t_ng_start)
        #t_ng_start = time.time()
        
        flat_grads_val = flat_grads_val.to(self.device)

        flat_grads_mat = torch.zeros(self.curr_N, self.num_params)

        for i in remainList:
          flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])
        flat_grads_mat = (-1.0 * self.eta * flat_grads_mat).to(self.device)

        #print("Flattening time trn:", time.time()-t_ng_start)
        #t_ng_start = time.time()

        target_new = [self.target for i in range(len(remainList))]
        
        while(numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf # value for current iteration (validation loss)
            bestId = -1 # element to pick
            #t_one_elem = time.time()

            #idxs_remain = list(remainSet)
            
            #t_ng_start = time.time()

            all_gains = torch.matmul(flat_grads_mat[remainList],torch.transpose(flat_grads_val,1,0))
            all_gains = all_gains + self.sum_score
            #print(torch.sum(all_gains,1))

            #print("Multiplying time :", time.time()-t_ng_start)
            #t_ng_start = time.time()

            '''losses = torch.zeros(len(remainList))
            for i in range(len(remainList)):
              losses[i] = -1 * self.loss(torch.unsqueeze(all_gains[i],0), torch.unsqueeze(self.target,0))'''

            #print(target)
            losses = -1 * self.loss_nored(all_gains, torch.tensor(target_new,dtype=torch.long).to(self.device))

            #print("Loss time :", time.time()-t_ng_start)
            #t_ng_start = time.time()
            
            # print(all_gains.shape)
            tmpid = torch.argmax(losses).item()
            bestId = remainList[tmpid]
            #bestGain = all_gains[tmpid]

            greedySet.add(bestId)
            target_new.remove(self.target)
            remainList.remove(bestId)            

            #print(tmpid)
            #print(bestId)
            #if self.target.item() == 1:
              #print(self.sum_score)
            #print(all_gains[tmpid])
            #t_ng_start = time.time()
            #self.grads_currX = self._update_gradients_subset(self.grads_currX, bestId, theta_init)
            grads_currX = self._update_gradients_subset(grads_currX, bestId, theta_init)    
            
            for val in range(self.no_cl):
                flat_grads_val[val] = self.flatten_params(self.grads_curr_subset[val])    
            flat_grads_val = flat_grads_val.to(self.device)

            #print("Flattening time trn:", time.time()-t_ng_start)
            #t_ng_start = time.time()    

            '''self.model.load_state_dict(theta_init)
            #self.model.zero_grad()
            with torch.no_grad(): # perform one-step update
                for i, param in enumerate(self.model.parameters()):                           
                    param.data.sub_(self.eta * self.grads_currX[i])

            floss = 0
            for i, data_i in  enumerate(self.train_loader, 0):
              inputs_i, target_i = data_i
              #inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
              scores = self.model(inputs_i)
              floss += self.loss(scores, target_i)
            print(floss/(i+1))'''     
            
            numSelected += 1
            #print(numSelected)
        #print("Naive greedy total time with taylor:", time.time()-t_ng_start)
        #print(self.sum_score)
        return list(greedySet)#, grads_currX

class SetFunctionTaylorDeep_ReLoss_Mean(object):
    
    def __init__(self, x_trn, y_trn, trn_batch, x_val, y_val,valid, model, loss_criterion, loss_nored, eta,device,N_trn): #trn_batch,

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
        self.N = N_trn


    def class_wise(self):  

      self.train_loader =[]
      for item in range(math.ceil(self.N /self.trn_batch)):
        inputs = self.x_trn[item*self.trn_batch:(item+1)*self.trn_batch]
        target = self.y_trn[item*self.trn_batch:(item+1)*self.trn_batch]
        self.train_loader.append((inputs,target))

      self.valid_loader =[]

      if self.valid:

        classes = torch.unique(self.y_val)

        self.no_cl = len(classes)

        for cl in classes:

          idx = (self.y_val == cl).nonzero().flatten()
          idx.tolist()

          inputs = self.x_val[idx]
          target = self.y_val[idx]
          self.valid_loader.append((inputs,target))   

      else:

        classes = torch.unique(self.y_trn)

        self.no_cl = len(classes)

        for cl in classes:

          idx = (self.y_trn == cl).nonzero().flatten()
          idx.tolist()

          inputs = self.x_trn[idx]
          target = self.y_trn[idx]
          self.valid_loader.append((inputs,target))         


    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        self.class_wise()
          
        losses =[]
        for i, data_i in  enumerate(self.train_loader, 0):
          inputs_i, target_i = data_i
          inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          losses.append(self.loss_nored(scores, target_i))

        losses = torch.cat(losses,0)

        grads_vec = [0 for _ in range(self.N)]   # zero is just a placeholder

        for name,_ in self.model.named_children():
            self.last_layer_name = name

        '''self.last_params =[]
        for name, param in self.model.named_parameters():
            if self.last_layer_name not in name :
                param.requires_grad = False
            else:
                self.last_params.append(param)'''

        for item in range(self.N):
            grads_vec[item] = torch.autograd.grad(losses[item],self.model.parameters() , retain_graph=True) #,self.last_params

        self.grads_per_elem = grads_vec


    def _compute_init_valloss_grads(self, theta_init):
        # Now compute the Validation loss initial gradient
        
        self.model.load_state_dict(theta_init)
        self.model.zero_grad() 

        self.sum_score = torch.zeros(self.no_cl,self.no_cl)
        self.sum_score = self.sum_score.to(self.device)
        self.target = []#torch.zeros(self.no_cl,dtype=torch.long)
        self.weight = []

        self.N_val = self.y_val.shape[0]
        #floss = 0
        for i, data_i in  enumerate(self.valid_loader, 0):
          inputs_i, target_i = data_i
          inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          self.sum_score[i] = torch.sum(scores, 0)
          self.sum_score[i] /= target_i.shape[0]
          self.target.append(target_i[0])
          self.weight.append(target_i.shape[0]*1.0/self.N_val)
          #self.sum_score[i] *= weight
          #floss -= self.weight[i]*self.loss(scores,target_i)

        #print("Complete",floss)
        grads_vec = [ [0 for _ in range(self.no_cl)]  for _ in range(self.no_cl)]   # zero is just a placeholder

        #floss = -1.0 * self.loss_nored(self.sum_score, torch.tensor(self.target))
        #print("Original",floss.sum()/4)

        #if floss.sum() >= 0:
        #  print(self.sum_score)

        for cl in range(self.no_cl):
          for val in range(self.no_cl):
            grads_vec[cl][val] = torch.autograd.grad(self.sum_score[cl][val], self.model.parameters(), retain_graph=True) #last_params

        self.grads_curr_subset  = grads_vec


    def _update_gradients_subset(self, grads_X, element, theta_init):
        
        if not grads_X:
            grads_X = list(copy.deepcopy(self.grads_per_elem[element]))
        else:
            grads_e = self.grads_per_elem[element]
            for i, _ in enumerate(self.model.parameters()): #self.last_params): 
                grads_X[i] += grads_e[i]

        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        with torch.no_grad(): # perform one-step update
            for i, param in enumerate(self.model.parameters()): #self.last_params): #                          
                param.data.sub_(self.eta * grads_X[i]) 
        

        self.sum_score = torch.zeros(self.no_cl,self.no_cl)
        self.sum_score = self.sum_score.to(self.device)
        #floss = 0
        for i, data_i in  enumerate(self.valid_loader, 0):
          inputs_i, target_i = data_i
          inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
          scores = self.model(inputs_i)
          #weight = target_i.shape[0]*1.0/self.N_val
          #floss -= self.weight[i]*self.loss(scores,target_i)
          self.sum_score[i] = torch.sum(scores, 0)
          self.sum_score[i] /= target_i.shape[0]
          
          #self.sum_score[i] *= weight

        #print("Complete",floss)
        grads_vec = [ [0 for _ in range(self.no_cl)]  for _ in range(self.no_cl)]   # zero is just a placeholder
        
        for cl in range(self.no_cl):
          for val in range(self.no_cl):
            grads_vec[cl][val] = torch.autograd.grad(self.sum_score[cl][val], self.model.parameters(), retain_graph=True) #self.last_params

        self.grads_curr_subset  = grads_vec

        return grads_X
        

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
        #remainSet = set(range(self.curr_N))
        if remainList == None:
          remainList = list(range(self.N))

        t_ng_start = time.time()    # naive greedy start time 

        self.num_params = self.flatten_params(self.grads_per_elem[0]).shape[0]

        #print(len(self.grads_per_elem[0]))
        #print(self.grads_per_elem[0].shape)

        flat_grads_val = torch.zeros(self.no_cl,self.no_cl, self.num_params)
        for cl in range(self.no_cl):
          for val in range(self.no_cl):
            flat_grads_val[cl][val] = self.flatten_params(self.grads_curr_subset[cl][val])  ## Total parameter size
        flat_grads_val = flat_grads_val.to(self.device)

        #print("Flattening time:", time.time()-t_ng_start)
        #t_ng_start = time.time()
        flat_grads_mat = torch.zeros(self.N, self.num_params)
        for i in remainList:
          flat_grads_mat[i] = self.flatten_params(self.grads_per_elem[i])
        flat_grads_mat = (-1.0 * self.eta* flat_grads_mat).to(self.device) #

        #print("Flattening time trn:", time.time()-t_ng_start)
        #t_ng_start = time.time()
        '''target_new = []
        for i in self.target: 
          for j in range(len(remainList)):
            target_new.append(i)'''

        target_new = torch.tensor(self.target).repeat_interleave(len(remainList)).to(self.device)
        
        while(numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf # value for current iteration (validation loss)
            bestId = -1 # element to pick
            
            #t_one_elem = time.time()
            #t_ng_start = time.time()
            #losses = torch.zeros(len(remainList))
            
            all_gains = torch.matmul(flat_grads_mat[remainList],torch.transpose(flat_grads_val,1,2))
            scores = self.sum_score.unsqueeze(1).expand(self.no_cl, len(remainList), self.no_cl)
            all_gains = all_gains + scores
            all_gains = torch.reshape(all_gains, (-1,self.no_cl))
            los_cal = -1.0 * self.loss_nored(all_gains, target_new)
            los_cal = torch.reshape(los_cal,(self.no_cl,len(remainList)))
            #losses =  los_cal.sum(0)
            losses = torch.matmul(torch.transpose(los_cal,0,1),torch.tensor(self.weight).to(self.device))
            #print(losses.shape)

            #print("Loss time :", time.time()-t_ng_start)
            #t_ng_start = time.time()
            # print(all_gains.shape)
            tmpid = torch.argmax(losses).item()
            bestId = remainList[tmpid]
            #print("Estimated",losses[tmpid],bestId)
            #bestGain = all_gains[tmpid]
            '''act_loss = -1.0 * self.loss_nored(self.sum_score,torch.tensor(self.target))
            print("Actu",act_loss, act_loss.sum()) 
            print("Esti",torch.transpose(los_cal,0,1)[tmpid],losses[tmpid],self.y_trn[bestId])
            print(tmpid,bestId)'''

            greedySet.add(bestId)
            '''for i in self.target:
              target_new.remove(i)'''
            remainList.remove(bestId)     
            target_new = torch.tensor(self.target).repeat_interleave(len(remainList)).to(self.device)       

            #if self.target.item() == 1:
              #print(self.sum_score)
            #print(all_gains[tmpid])
            #t_ng_start = time.time()
            grads_currX = self._update_gradients_subset(grads_currX, bestId, theta_init)    
            
            for cl in range(self.no_cl):
              for val in range(self.no_cl):
                flat_grads_val[cl][val] = self.flatten_params(self.grads_curr_subset[cl][val])    
            flat_grads_val = flat_grads_val.to(self.device)

            #print("Flattening time trn:", time.time()-t_ng_start)
            #t_ng_start = time.time()          
            
            numSelected += 1
            #print(numSelected)
        print("Naive greedy total time with taylor:", time.time()-t_ng_start)
        #print(self.sum_score)
        return list(greedySet)#, grads_currX