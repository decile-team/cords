import torch
from torch import nn
from cords.utils.utils import dummy_context


# LSTM for text classification
class LSTMModel(nn.Module):

    def __init__(self, vocab_size, hidden_units, num_layers, embed_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_units, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, num_classes)
        self.num_classes = num_classes
        self.feature_dim = hidden_units

    def forward(self, text, offsets, last=False, freeze=False):
        with torch.no_grad() if freeze else dummy_context():
            embedded = self.embedding(text, offsets)
            h = self.lstm(embedded)
        scores = self.fc(h)
        if last:
            return scores, embedded
        else:
            return scores

    def get_feature_dim(self):
        return self.feature_dim

    def get_embedding_dim(self):
        return self.feature_dim
