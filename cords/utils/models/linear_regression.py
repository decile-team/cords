import torch
import torch.nn as nn

class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNet, self).__init__()
        self.linear = nn.Linear(input_dim, 1)#,bias=False)
        self.feature_dim = input_dim
    
    def forward(self, x, last=False):
        scores = self.linear(x)
        #scores = torch.sigmoid(self.linear(x))
        #return scores.view(-1)
        if last:
            return scores.view(-1), x
        else:
            return scores.view(-1)

    def get_embedding_dim(self):
        return self.feature_dim

class LogisticNet(nn.Module):
    def __init__(self, input_dim):#,num_cls):
        super(LogisticNet, self).__init__()
        self.linear = nn.Linear(input_dim,1)
        #self.feature_dim = input_dim
    
    def forward(self, x, last=False):
        scores = self.linear(x)
        scores = torch.sigmoid(self.linear(x).view(-1))
        return scores

class DualNet(nn.Module):
    def __init__(self, input_dim):
        super(DualNet, self).__init__()
        self.linear = nn.Linear(input_dim, 1,bias=False)
    
    def forward(self, x):
        scores = self.linear(x)
        return scores.view(-1)