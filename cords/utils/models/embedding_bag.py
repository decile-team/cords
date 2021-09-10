import torch
from torch import nn
from cords.utils.utils import dummy_context


class EmbeddingBagModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_classes):
        super(EmbeddingBagModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_classes)
        # self.init_weights()
        self.num_classes = num_classes
        self.feature_dim = embed_dim

    # def init_weights(self):
    #     initrange = 0.5
    #     self.embedding.weight.data.uniform_(-initrange, initrange)
    #     self.fc.weight.data.uniform_(-initrange, initrange)
    #     self.fc.bias.data.zero_()

    def forward(self, text, offsets, last=False, freeze=False):
        with torch.no_grad() if freeze else dummy_context():
            embedded = self.embedding(text, offsets)
        scores = self.fc(embedded)
        if last:
            return scores, embedded
        else:
            return scores

    def get_feature_dim(self):
        return self.feature_dim

    def get_embedding_dim(self):
        return self.feature_dim
