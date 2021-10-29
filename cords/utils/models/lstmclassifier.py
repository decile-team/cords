import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from cords.utils.data.datasets.SL.builder import loadGloveModel

class LSTMClassifier(nn.Module):
    def __init__(self, num_classes, wordvec_dim, weight_path, num_layers=1, hidden_size=150):
        super(LSTMClassifier, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.embedding_length = wordvec_dim 
        weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
        wordvec = loadGloveModel(weight_full_path)
        weight = torch.tensor(wordvec.values, dtype=torch.float)  # word embedding for the embedding layer
        
        self.embedding = nn.Embedding(
            weight.shape[0], self.embedding_length)  # Embedding layer
        self.embedding = self.embedding.from_pretrained(
            weight, freeze=False)  # Load pretrianed word embedding, and fine-tuing
        self.lstm = nn.LSTM(self.embedding_length,
                            self.hidden_size, num_layers=num_layers, batch_first=True)  # lstm
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input_sentence, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                x = self.embedding(input_sentence)  # (batch_size, batch_dim, embedding_length)
                output, (final_hidden_state, final_cell_state) = self.lstm(x)
        else:
            x = self.embedding(input_sentence)  # (batch_size, batch_dim, embedding_length)
            output, (final_hidden_state, final_cell_state) = self.lstm(x)
        logits = self.fc(final_hidden_state[-1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & logits.size() = (batch_size, num_classes)
        if last:
            return logits, final_hidden_state[-1]
        else:
            return logits

    def get_feature_dim(self):
        return self.hidden_size

    def get_embedding_dim(self):
        return self.hidden_size


