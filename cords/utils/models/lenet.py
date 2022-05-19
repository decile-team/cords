'''LeNet in PyTorch.'''


import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.embDim = 84
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)


    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = F.relu(self.conv1(x))
                out = F.max_pool2d(out, 2)
                out = F.relu(self.conv2(out))
                out = F.max_pool2d(out, 2)
                out = out.view(out.size(0), -1)
                out = F.relu(self.fc1(out))
                e = F.relu(self.fc2(out))
        else:
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            e = F.relu(self.fc2(out))
        out = self.fc3(e)
        if last:
            return out, e
        else:
            return out


    def get_embedding_dim(self):
        return self.embDim
