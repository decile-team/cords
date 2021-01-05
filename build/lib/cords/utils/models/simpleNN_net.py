import torch.nn as nn
import torch.nn.functional as F


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_units):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, num_classes)
        self.feature_dim = hidden_units
    
    def forward(self, x, last=False):
        l1scores = F.relu(self.linear1(x))
        scores = self.linear2(l1scores)
        if last:
            return scores, l1scores
        else:
            return scores

    def get_feature_dim(self):
        return self.feature_dim

    def get_embedding_dim(self):
        return self.feature_dim


class ThreeLayerNet(nn.Module):
    def __init__(self, input_dim, num_classes, h1, h2):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, num_classes)
        self.feature_dim = h2
    
    def forward(self, x, last=False):
        l1scores = F.relu(self.linear1(x))
        l2scores = F.relu(self.linear2(l1scores))
        scores = self.linear3(l2scores)
        if last:
            return scores, l2scores
        else:
            return scores

    def get_feature_dim(self):
        return self.feature_dim

    def get_embedding_dim(self):
        return self.feature_dim