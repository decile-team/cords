import json

import torch, pickle
from torch import nn

from cords.utils.models.lstm import LSTM
from cords.utils.utils import dummy_context


# LSTM for text classification
class LSTMModel(nn.Module):

    def __init__(self, vocab_size, hidden_units, num_layers, embed_dim, num_classes, pretrained_embedding=None,
                 dataset=None):
        super(LSTMModel, self).__init__()
        print("vocab_size: %s" % vocab_size)
        print("vocab_size: %s, embed_dim: %s" % (vocab_size, embed_dim))
        # import pdb; pdb.set_trace()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_units, num_layers=num_layers, batch_first=True)
        self.lstm = LSTM(input_size=embed_dim, hidden_size=hidden_units)
        self.fc = nn.Linear(hidden_units, num_classes)
        self.num_classes = num_classes
        self.feature_dim = hidden_units
        self.pretrained_embedding = pretrained_embedding

        if pretrained_embedding is not None:
            self.load_pretrained(pretrained_embedding, dataset)

    def load_pretrained(self, pretrained_embedding, dataset):
        print("Loading pretrained embedding: %s with dataset: %s. " % (pretrained_embedding, dataset))

        if pretrained_embedding == "glove_twitter_25":
            glove_vocab_txt_path = "./data/glove.twitter/glove.twitter.27B.25d.txt"
        elif pretrained_embedding == "glove_twitter_50":
            glove_vocab_txt_path = "./data/glove.twitter/glove.twitter.27B.50d.txt"
        elif pretrained_embedding == "glove_twitter_100":
            glove_vocab_txt_path = "./data/glove.twitter/glove.twitter.27B.100d.txt"
        elif pretrained_embedding == "glove_twitter_200":
            glove_vocab_txt_path = "./data/glove.twitter/glove.twitter.27B.200d.txt"
        elif pretrained_embedding == "glove.6B.50d":
            glove_vocab_txt_path = "./data/glive.6b/glove.6B.50d.txt"
        elif pretrained_embedding == "glove.6B.100d":
            glove_vocab_txt_path = "./data/glive.6b/glove.6B.100d.txt"
        elif pretrained_embedding == "glove.6B.200d":
            glove_vocab_txt_path = "./data/glive.6b/glove.6B.200d.txt"
        elif pretrained_embedding == "glove.6B.300d":
            glove_vocab_txt_path = "./data/glive.6b/glove.6B.300d.txt"
        else:
            raise Exception("pretrained_embedding %s does not exist. " % pretrained_embedding)

        if dataset == "corona":
            dataset_vocab_path = "./data/corona_vocab.pickle"
        elif dataset == "twitter":
            dataset_vocab_path = "./data/twitter_vocab.pickle"
        elif dataset == "news":
            dataset_vocab_path = "./data/news_vocab.pickle"
        elif dataset == "ag_news":
            dataset_vocab_path = "./data/ag_news_vocab.pickle"
        else:
            raise Exception("Dataset %s does not exist. " % dataset)

        with open(dataset_vocab_path, "rb") as handle:
            dataset_vocab = pickle.load(handle)

        mapped_line = 0
        glove_vocab_txt = open(glove_vocab_txt_path, "r")
        for line in glove_vocab_txt.readlines():
            line = line.split()
            word = line[0]
            if word in dataset_vocab:
                # index 100815 is out of bounds for dimension 0 with size 41302
                # print("word: %s. " % word)
                # print("dataset_vocab[word]: %s" % dataset_vocab[word])
                # print("len(dataset_vocab): %s" % len(dataset_vocab))
                self.embedding.weight.data[dataset_vocab[word]] = torch.tensor([float(w) for w in line[1:]]).float()
                mapped_line += 1

        print("Pretrained embedding loaded, mapped line: %s. " % mapped_line)

    def forward(self, text, last=False, freeze=False):
        with torch.no_grad() if freeze else dummy_context():
            embedded = self.embedding(text)
            h = self.lstm(embedded, freeze=freeze)
        scores = self.fc(h)
        if last:
            return scores, h
        else:
            return scores

    def get_feature_dim(self):
        return self.feature_dim

    def get_embedding_dim(self):
        return self.feature_dim
