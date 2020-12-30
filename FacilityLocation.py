import numpy as np
import torch

from queue import PriorityQueue

class SetFunctionFacLoc(object):

    def __init__(self, device ,train_full_loader):#, valid_loader): 
        
        self.train_loader = train_full_loader      
        self.device = device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def distance(self,x, y, exp = 2):

      n = x.size(0)
      m = y.size(0)
      d = x.size(1)

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

          self.N +=  inputs_i.size()[0] 

          g_is.append(inputs_i)


        self.sim_mat = torch.zeros([self.N, self.N],dtype=torch.float32)

        first_i = True

        for i, g_i in  enumerate(g_is, 0):

          if first_i:
            size_b = g_i.size(0)
            first_i = False

          for j, g_j in  enumerate(g_is, 0):

            self.sim_mat[i*size_b: i*size_b + g_i.size(0) ,j*size_b: j*size_b + g_j.size(0)] = self.distance(g_i, g_j)

      dist = self.sim_mat.sum(1)

      self.const = torch.max(dist).item()
      self.sim_mat = self.const - self.sim_mat

      
      bestId = torch.argmax(self.N*self.const - dist).item()

      self.max_sim = self.sim_mat[bestId].to(self.device)

      return bestId
    

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
            gain = (torch.max(self.max_sim ,self.sim_mat[i].to(self.device)) - self.max_sim).sum()#L([0]+greedyList+[i]) - L([0]+greedyList)  #s_0 = self.x_trn[0]
            if bestGain < gain.item():
              bestGain = gain.item()
              bestId = i

          greedyList.append(bestId)
          not_selected.remove(bestId)
          numSelected += 1

          self.max_sim = torch.max(self.max_sim,self.sim_mat[bestId].to(self.device))#.to(self.device))

      return greedyList

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
      greedyList = [id_first,second[1]]
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

              curr_gain = (torch.max(self.max_sim ,self.sim_mat[first[1]].to(self.device)) - self.max_sim).sum()
              self.gains.put((-curr_gain.item(), first[1]))


              if curr_gain.item() >= bestGain:
                  
                bestGain = curr_gain.item()
                bestId = first[1]

          #print(bestGain,end=",")
          greedyList.append(bestId)
          numSelected += 1

          self.max_sim = torch.max(self.max_sim,self.sim_mat[bestId].to(self.device))

      #print()
      #gamma = self.compute_gamma(greedyList)

      return greedyList
