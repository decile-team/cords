import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, n_out):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1),
                                        nn.MaxPool2d(3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 32, 3, padding=1),
                                        nn.MaxPool2d(3, stride=2, padding=1),
                                        nn.ReLU()
                                        )
        self.dropout = nn.Dropout(p=0.5)
        self.dense = nn.Linear(32 * 7 * 7, n_out)
        self.embDim = 32*7*7

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag

    def get_embedding_dim(self):
        return self.embDim

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                x = self.conv(x)
                x = x.view(-1, 32 * 7 * 7)
                e = self.dropout(x)
        else:
            x = self.conv(x)
            x = x.view(-1, 32 * 7 * 7)
            e = self.dropout(x)
        x = self.dense(e)
        if last:
            return x, e
        else:
            return x