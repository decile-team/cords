# Please run this script at the root dir of cords
import json
import pickle

import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import spacy
# from sklearn.externals import joblib
import joblib


def build_vocab(texts, tokenizer):
    counter = Counter()
    d_index = {}
    for i, text in enumerate(texts):
        tokenized_text = tokenizer(text)
        for word in tokenized_text:
            if word not in d_index:
                d_index[word] = len(d_index) + 1
        counter.update(tokenized_text)
    return Vocab(counter), d_index


# def split_X_y(train, valid, test, y_ind):
#     X_train, X_valid, X_test = train.drop(train.columns[y_ind], axis=1), \
#                                valid.drop(valid.columns[y_ind], axis=1), \
#                                test.drop(test.columns[y_ind], axis=1),
#     y_train, y_valid, y_test = train.iloc[:, y_ind], valid.iloc[:, y_ind], test.iloc[:, y_ind]
#     return X_train, X_valid, X_test, y_train, y_valid, y_test


def split_X_y(data, y_ind):
    X = data.drop(data.columns[y_ind], axis=1)
    y = data.iloc[:, y_ind]
    return X, y


def split_train_valid_test(X, y, r_train, r_valid):
    l = X.shape[0]
    i_train, i_valid = int(l * r_train), int(l * (r_train + r_valid))
    indices = np.random.permutation(X.shape[0])
    train_indices, valid_indices, test_indices = indices[: i_train], indices[i_train: i_valid], indices[i_valid:]
    return X.iloc[train_indices], X.iloc[valid_indices], X.iloc[test_indices], y[train_indices], y[valid_indices], y[
        test_indices]


def normalize_numeric_(data):
    for column_name in data.columns:
        column = data[column_name]
        if column.dtype in ["int64", "float64"]:
            # Gaussianize
            column = (column - column.mean()) / column.std()
            column_max, column_min = column.max(), column.min()
            if column_max == column_min:
                raise Exception("Column %s has all same values. " % column_name)
            else:
                # Map to 0 to 1
                data[column_name] = (column - column_min) / (column_max - column_min)


# Fill nans with median of training data
def fill_nan(X_train, X_valid, X_test):
    for column_name in X_train.columns:
        column_train, column_valid, column_test = X_train[column_name], X_valid[column_name], X_test[column_name]
        if column_train.dtype in ["int64", "float64"]:
            train_median = column_train.median()
            X_train[column_name] = column_train.replace(np.nan, train_median)
            X_valid[column_name] = column_valid.replace(np.nan, train_median)
            X_test[column_name] = column_test.replace(np.nan, train_median)


spacy.load('en_core_web_sm')


def process_and_save_data(filename, data, y_ind, text, r_train=0.7, r_valid=0.1, r_test=0.2):
    assert r_train + r_valid + r_test == 1, "Ratio should sum up to one. "
    assert y_ind < 0, "Please negative indexing for y. "
    data.fillna('', inplace=True)
    # split X and y
    X, y = split_X_y(data, y_ind)
    # Factorize y
    y = pd.factorize(y)[0]
    print("unique y: %s. " % np.unique(y))
    if text:
        texts = X.iloc[:, 0]
        texts = [text.lower() for text in texts]
        tokenizer = get_tokenizer('spacy', language='en')
        vocab, d_index = build_vocab(texts, tokenizer)
        tokenized_index = []
        for text in texts:
            _tokenized_index = torch.tensor([d_index[token] for token in tokenizer(text)], dtype=torch.long)
            tokenized_index.append(_tokenized_index)

        # Test tensors
        d = {}
        for _tokenized_index in tokenized_index:
            for index in _tokenized_index:
                index = index.item()
                if index not in d:
                    d[index] = 0
                d[index] += 1

        for word in d_index:
            assert vocab[word] == d[d_index[word]]

        # Truncate data length to 0.9 quantile (reject long outliers), then padding 0
        max_len = int(
            np.quantile(np.sort([_tokenized_index.size(0) for _tokenized_index in tokenized_index])[::-1], 0.9))
        tokenized_index = [_tokenized_index[:max_len] for _tokenized_index in tokenized_index]

        X = pad_sequence(tokenized_index, batch_first=True).detach().numpy()
        X = pd.DataFrame(X, dtype=np.long)
    else:
        # Normalize X
        normalize_numeric_(X)
        X = pd.get_dummies(X)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_train_valid_test(X, y, r_train, r_valid)

    fill_nan(X_train, X_valid, X_test)

    n_classes = np.unique(y).size

    if text:
        # Index starts from 1, vocab_size should be len(d_index)+1
        input_dim = len(d_index) + 1
        # Pandas to numpy
        X_train, X_valid, X_test = \
            X_train.to_numpy().astype("long"), \
            X_valid.to_numpy().astype("long"), \
            X_test.to_numpy().astype("long")
    else:
        input_dim = X_train.shape[1]
        X_train, X_valid, X_test = \
            X_train.to_numpy().astype("float32"), \
            X_valid.to_numpy().astype("float32"), \
            X_test.to_numpy().astype("float32")
    train = [(x, _y) for (x, _y) in zip(X_train, y_train)]
    valid = [(x, _y) for (x, _y) in zip(X_valid, y_valid)]
    test = [(x, _y) for (x, _y) in zip(X_test, y_test)]

    if text:
        with open(os.path.join("data", "%s_vocab.pickle" % filename), "wb") as handle:
            pickle.dump(d_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(vocab, handle, protocol=pickle.DEFAULT_PROTOCOL)
        # joblib.dump(vocab, os.path.join("data", "%s_vocab.pickle" % filename))
        # torch.save(vocab, os.path.join("data", "%s_vocab.pickle" % filename))
        # with open(os.path.join("data", "%s_vocab.pickle" % filename), "w") as f:
        #     json.dump(d_index, f)

    with open(os.path.join("data", "%s.pickle" % filename), "wb") as handle:
        # pickle.dump((train, valid, test, input_dim, n_classes), handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump((train, valid, test, input_dim, n_classes), handle, protocol=pickle.DEFAULT_PROTOCOL)


# Train: 0.7 / valid: 0.1 / test: 0.2
if __name__ == "__main__":
    np.random.seed(9527)

    # # # NLP
    # # ## Corona
    # # corona_train = pd.read_csv("./data/NLP/corona/Corona_NLP_train.csv", usecols=["OriginalTweet", "Sentiment"],
    # #                            encoding='ISO-8859-1')
    # # corona_test = pd.read_csv("./data/NLP/corona/Corona_NLP_test.csv", usecols=["OriginalTweet", "Sentiment"],
    # #                           encoding='ISO-8859-1')
    # # corona = pd.concat([corona_train, corona_test])
    # # corona = corona.sample(frac=1).reset_index(drop=True)
    # # process_and_save_data("corona", corona, -1, True, r_train=0.7, r_valid=0.1, r_test=0.2)
    # #
    # # # [41157 rows x 2 columns]
    # # print("corona_train: \n", corona_train)
    # # # [3798 rows x 2 columns]
    # # print("corona_test: \n", corona_test)
    # # print("corona: \n", corona)
    #
    # ## News
    # news_fake = pd.read_csv("./data/NLP/news/Fake.csv", usecols=["title", "text"])
    # news_fake["fake"] = 1
    # news_fake["text"] = news_fake["title"] + '\n' + news_fake["text"]
    # news_fake = news_fake.drop(columns=['title'])
    # news_true = pd.read_csv("./data/NLP/news/True.csv", usecols=["title", "text"])
    # news_true["fake"] = 0
    # news_true["text"] = news_true["title"] + '\n' + news_true["text"]
    # news_true = news_true.drop(columns=['title'])
    # news = pd.concat([news_fake, news_true])
    # news = news.sample(frac=1).reset_index(drop=True)
    # process_and_save_data("news", news, -1, True, r_train=0.7, r_valid=0.1, r_test=0.2)
    #
    # # [23481 rows x 4 columns]
    # print("news_fake: \n", news_fake)
    # # [21417 rows x 4 columns]
    # print("news_true: \n", news_true)
    # # [44898 rows x 4 columns]
    # print("news: \n", news)
    #
    # ## Twitter
    # twitter_train = pd.read_csv("./data/NLP/twitter/twitter_training.csv", usecols=[3, 2])
    # twitter_train.columns = ["Sentiment", "text"]
    # twitter_test = pd.read_csv("./data/NLP/twitter/twitter_validation.csv", usecols=[3, 2])
    # twitter_test.columns = ["Sentiment", "text"]
    # twitter = pd.concat([twitter_train, twitter_test])
    # # twitter = twitter.sample(frac=1).reset_index(drop=True)
    # columns_titles = ["text", "Sentiment"]
    # twitter = twitter.reindex(columns=columns_titles)
    # process_and_save_data("twitter", twitter, -1, True, r_train=0.7, r_valid=0.1, r_test=0.2)
    #
    # # [74681 rows x 4 columns]
    # print("twitter_train: \n", twitter_train)
    # # [999 rows x 4 columns]
    # print("twitter_test: \n", twitter_test)
    # print("twitter: \n", twitter)
    #
    # # Tabular
    # ## Airline
    # airline_train = pd.read_csv("./data/TABULAR/airline/train.csv")
    # airline_test = pd.read_csv("./data/TABULAR/airline/test.csv")
    # airline_train = airline_train.drop(airline_train.columns[[0, 1]], axis=1)
    # airline_test = airline_test.drop(airline_test.columns[[0, 1]], axis=1)
    # airline = pd.concat([airline_train, airline_test])
    # airline = airline.sample(frac=1).reset_index(drop=True)
    # process_and_save_data("airline", airline, -1, False, r_train=0.7, r_valid=0.1, r_test=0.2)
    #
    # # [103904 rows x 25 columns]
    # print("airline_train: \n", airline_train)
    # # [25976 rows x 25 columns]
    # print("airline_test: \n", airline_test)
    # print("airline: \n", airline)
    #
    # ## Loan
    # loan_train = pd.read_csv("./data/TABULAR/loan/Training Data.csv")
    # loan_test = pd.read_csv("./data/TABULAR/loan/Test Data.csv")
    # loan_train = loan_train.drop(columns=["Id"])
    # loan = loan_train
    # loan = loan.sample(frac=1).reset_index(drop=True)
    # process_and_save_data("loan", loan, -1, False, r_train=0.7, r_valid=0.1, r_test=0.2)
    #
    # # [252000 rows x 13 columns]
    # print("loan_train: \n", loan_train)
    # # [28000 rows x 12 columns]
    # print("loan_test: \n", loan_test)
    # print("loan: \n", loan)
    #
    # ## Olympic
    # # [271116 rows x 15 columns]
    # olympic = pd.read_csv("./data/TABULAR/olympic/athlete_events.csv")
    # olympic = olympic.drop(columns=["ID", "Name"])
    # olympic = olympic.sample(frac=1).reset_index(drop=True)
    # process_and_save_data("olympic", olympic, -1, False, r_train=0.7, r_valid=0.1, r_test=0.2)
    #
    # print("olympic: \n", olympic)

    # # Extra NLP: amazon
    # ## Amazon
    # amazon_label, amazon_text = [], []
    # with open("./data/NLP/amazon/train.ft.txt") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         label, text = line.split(" ", 1)
    #         if label == "__label__2":
    #             amazon_label.append("1")
    #         elif label == "__label__1":
    #             amazon_label.append("0")
    #         else:
    #             raise Exception("Label error: label %s does not exist. " % label)
    #         amazon_text.append(text)
    #
    # with open("./data/NLP/amazon/test.ft.txt") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         label, text = line.split(" ", 1)
    #         if label == "__label__2":
    #             amazon_label.append("1")
    #         elif label == "__label__1":
    #             amazon_label.append("0")
    #         else:
    #             raise Exception("Label error: label %s does not exist. " % label)
    #         amazon_text.append(text)
    #
    # amazon = np.stack((amazon_text, amazon_label)).T
    # amazon = pd.DataFrame(amazon)
    # amazon = amazon.sample(frac=1).reset_index(drop=True)
    # columns_titles = ["text", "Sentiment"]
    # amazon = amazon.reindex(columns=columns_titles)
    # print(amazon.shape)
    # process_and_save_data("amazon", amazon, -1, True, r_train=0.7, r_valid=0.1, r_test=0.2)

    # # Extra NLP: spam
    # spam = pd.read_csv("./data/NLP/spam/spam.csv")
    # spam = spam.sample(frac=1).reset_index(drop=True)
    # print(1234)

    # Extra NLP: ag_news
    # ag_news.columns
    # Index(['Class Index', 'Title', 'Description'], dtype='object')
    ag_news_train = pd.read_csv("./data/NLP/ag_news/train.csv")
    ag_news_test = pd.read_csv("./data/NLP/ag_news/test.csv")
    ag_news = pd.concat([ag_news_train, ag_news_test])
    ag_news_title = ag_news["Title"]
    ag_news_class = ag_news["Class Index"]
    ag_news_description = ag_news["Description"]

    ag_news = ag_news.drop(ag_news.columns[-1], axis=1)
    ag_news_text = ag_news_title + ag_news_description

    ag_news["Title"] = ag_news_text.str.lower()
    ag_news = ag_news.reindex(columns=["Title", "Class Index"])

    process_and_save_data("ag_news", ag_news, -1, True, r_train=0.7, r_valid=0.1, r_test=0.2)

