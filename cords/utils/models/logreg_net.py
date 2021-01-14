import torch.nn as nn


#from sklearn import datasets
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#bc = datasets.load_breast_cancer()
#x, y = bc.data, bc.target
#
#x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=0.3)
#sc = StandardScaler()
#x_trn = sc.fit_transform(x_trn)
#x_tst = sc.transform(x_tst)
#
#x_trn = torch.from_numpy(x_trn.astype(np.float32))
#x_tst = torch.from_numpy(x_tst.astype(np.float32))
#y_trn = torch.from_numpy(y_trn.astype(np.int64))
#y_tst = torch.from_numpy(y_tst.astype(np.int64))
#N, M = x_trn.shape
#

### Logisitic Regression model
### The softmax will be applied by the torch's CrossEntropyLoss loss function
### Similar to that of a neural network pre-final layer scores.
class LogisticRegNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegNet, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.feature_dim = input_dim


    def forward(self, x, last=False):
        scores = self.linear(x)
        if last:
            return scores, x
        else:
            return scores


    def get_embedding_dim(self):
        return self.feature_dim
