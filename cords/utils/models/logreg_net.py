import torch.nn as nn
import torch

### Logisitic Regression model
### The softmax will be applied by the torch's CrossEntropyLoss loss function
### Similar to that of a neural network pre-final layer scores.
class LogisticRegNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegNet, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.feature_dim = input_dim


    def forward(self, x, last=False, freeze=False):
        scores = self.linear(x)
        if last:
            return scores, x
        else:
            return scores


    def get_embedding_dim(self):
        return self.feature_dim


class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNet, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # ,bias=False)
        self.feature_dim = input_dim

    def forward(self, x, last=False,freeze=False):

        scores = self.linear(x)

        if last:
            return scores, x
        else:
            return scores#.view(-1)

    def get_embedding_dim(self):
        return self.feature_dim