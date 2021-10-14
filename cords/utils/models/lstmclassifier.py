import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class LSTMClassifier(nn.Module):
    def __init__(self, config):
        super(LSTMClassifier, self).__init__()
        self.label_num = config.label_num
        self.hidden_size = 150
        self.embedding_length = config.wordvec_dim 
        self.word_embeddings = nn.Embedding(
            config.weight.shape[0], self.embedding_length)  # Embedding layer
        self.word_embeddings = self.word_embeddings.from_pretrained(
            config.weight, freeze=False)  # Load pretrianed word embedding, and fine-tuing
        self.lstm = nn.LSTM(self.embedding_length,
                            self.hidden_size, batch_first=True)  # lstm
        self.fc = nn.Linear(self.hidden_size, self.label_num)

    def forward(self, input_sentence, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                x = self.word_embeddings(input_sentence)  # (batch_size, batch_dim, embedding_length)
                output, (final_hidden_state, final_cell_state) = self.lstm(x)
        else:
            x = self.word_embeddings(input_sentence)  # (batch_size, batch_dim, embedding_length)
            output, (final_hidden_state, final_cell_state) = self.lstm(x)
        logits = self.fc(final_hidden_state[-1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & logits.size() = (batch_size, label_num)
        if last:
            return logits, final_hidden_state[-1]
        else:
            return logits


