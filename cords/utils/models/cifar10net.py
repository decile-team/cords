import torch.nn as nn
import torch.nn.functional as F
import torch

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.embDim = 256
        
        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)


    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = self.pool(F.relu(self.conv1(x)))
                out = self.pool(F.relu(self.conv2(out)))
                out = self.pool(F.relu(self.conv3(out)))
                out = out.view(-1, 64 * 4 * 4)
                out = F.relu(self.fc1(out))
                e = F.relu(self.fc2(out))
        else:
            out = self.pool(F.relu(self.conv1(x)))
            out = self.pool(F.relu(self.conv2(out)))
            out = self.pool(F.relu(self.conv3(out)))
            out = out.view(-1, 64 * 4 * 4)
            out = F.relu(self.fc1(out))
            e = F.relu(self.fc2(out))
        out = self.fc3(e)
        if last:
            return out, e
        else:
            return out


    def get_embedding_dim(self):
        return self.embDim
