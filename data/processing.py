# Please run this script at the root dir of cords

import pandas as pd
import numpy as np
import pickle


def split_X_y(train, valid, test, y_ind):
    X_train, X_valid, X_test = train.drop(train.columns[y_ind], axis=1), \
                               valid.drop(valid.columns[y_ind], axis=1), \
                               test.drop(test.columns[y_ind], axis=1),
    y_train, y_valid, y_test = train.iloc[:, y_ind], valid.iloc[:, y_ind], test.iloc[:, y_ind]
    return X_train, X_valid, X_test, y_train, y_valid, y_test


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


def process_and_save_data(filename, data, y_ind, text, r_train=0.7, r_valid=0.1, r_test=0.2):
    assert r_train + r_valid + r_test == 1, "Ratio should sum up to one. "
    # Factorize y
    data.iloc[:, y_ind] = pd.factorize(data.iloc[:, y_ind])[0]
    # Split dataloader
    train, valid, test = np.split(data.sample(frac=1),
                                  [int(r_train * len(data)), int((r_train + r_valid) * len(data))])
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_X_y(train, valid, test, y_ind)
    fill_nan(X_train, X_valid, X_test)
    if not text:
        # Normalize numerical dataloader
        normalize_numeric_(X_train)
        normalize_numeric_(X_valid)
        normalize_numeric_(X_test)
        # Transfer to dummy variable if not text dataloader
        X_train = pd.get_dummies(X_train)
        X_valid = pd.get_dummies(X_valid)
        X_test = pd.get_dummies(X_test)
    n_classes = np.unique(np.concatenate([y_train, y_valid, y_test])).size

    # Pandas to numpy
    X_train, X_valid, X_test = X_train.to_numpy(), X_valid.to_numpy(), X_test.to_numpy()
    input_dim = X_train.shape[1]
    train = [(x, _y) for (x, _y) in zip(X_train, y_train)]
    valid = [(x, _y) for (x, _y) in zip(X_valid, y_valid)]
    test = [(x, _y) for (x, _y) in zip(X_test, y_test)]

    with open('%s.pickle' % filename, 'wb') as handle:
        pickle.dump((train, valid, test, input_dim, n_classes), handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


# Train: 0.7 / valid: 0.1 / test: 0.2
if __name__ == "__main__":
    np.random.seed(9527)

    # NLP
    ## Corona
    corona_train = pd.read_csv("./data/NLP/corona/Corona_NLP_train.csv", usecols=["OriginalTweet", "Sentiment"],
                               encoding='ISO-8859-1')
    corona_test = pd.read_csv("./data/NLP/corona/Corona_NLP_test.csv", usecols=["OriginalTweet", "Sentiment"],
                              encoding='ISO-8859-1')
    corona = pd.concat([corona_train, corona_test])
    corona = corona.sample(frac=1).reset_index(drop=True)
    process_and_save_data("corona", corona, -1, True, r_train=0.7, r_valid=0.1, r_test=0.2)

    # [41157 rows x 2 columns]
    print("corona_train: \n", corona_train)
    # [3798 rows x 2 columns]
    print("corona_test: \n", corona_test)
    print("corona: \n", corona)

    ## News
    news_fake = pd.read_csv("./data/NLP/news/Fake.csv", usecols=["title", "text"])
    news_fake["fake"] = 1
    news_fake["text"] = news_fake["title"] + '\n' + news_fake["text"]
    news_fake = news_fake.drop(columns=['title'])
    news_true = pd.read_csv("./data/NLP/news/True.csv", usecols=["title", "text"])
    news_true["fake"] = 0
    news_true["text"] = news_true["title"] + '\n' + news_true["text"]
    news_true = news_true.drop(columns=['title'])
    news = pd.concat([news_fake, news_true])
    news = news.sample(frac=1).reset_index(drop=True)
    process_and_save_data("news", news, -1, True, r_train=0.7, r_valid=0.1, r_test=0.2)

    # [23481 rows x 4 columns]
    print("news_fake: \n", news_fake)
    # [21417 rows x 4 columns]
    print("news_true: \n", news_true)
    # [44898 rows x 4 columns]
    print("news: \n", news)

    ## Twitter
    twitter_train = pd.read_csv("./data/NLP/twitter/twitter_training.csv", usecols=[3, 2])
    twitter_train.columns = ["Sentiment", "text"]
    twitter_test = pd.read_csv("./data/NLP/twitter/twitter_validation.csv", usecols=[3, 2])
    twitter_test.columns = ["Sentiment", "text"]
    twitter = pd.concat([twitter_train, twitter_test])
    twitter = twitter.sample(frac=1).reset_index(drop=True)
    process_and_save_data("twitter", twitter, 0, True, r_train=0.7, r_valid=0.1, r_test=0.2)

    # [74681 rows x 4 columns]
    print("twitter_train: \n", twitter_train)
    # [999 rows x 4 columns]
    print("twitter_test: \n", twitter_test)
    print("twitter: \n", twitter)

    # Tabular
    ## Airline
    airline_train = pd.read_csv("./data/TABULAR/airline/train.csv")
    airline_test = pd.read_csv("./data/TABULAR/airline/test.csv")
    airline_train = airline_train.drop(airline_train.columns[[0, 1]], axis=1)
    airline_test = airline_test.drop(airline_test.columns[[0, 1]], axis=1)
    airline = pd.concat([airline_train, airline_test])
    airline = airline.sample(frac=1).reset_index(drop=True)
    process_and_save_data("airline", airline, -1, False, r_train=0.7, r_valid=0.1, r_test=0.2)

    # [103904 rows x 25 columns]
    print("airline_train: \n", airline_train)
    # [25976 rows x 25 columns]
    print("airline_test: \n", airline_test)
    print("airline: \n", airline)

    ## Loan
    loan_train = pd.read_csv("./data/TABULAR/loan/Training Data.csv")
    loan_test = pd.read_csv("./data/TABULAR/loan/Test Data.csv")
    loan_train = loan_train.drop(columns=["Id"])
    loan = loan_train
    loan = loan.sample(frac=1).reset_index(drop=True)
    process_and_save_data("loan", loan, -1, False, r_train=0.7, r_valid=0.1, r_test=0.2)

    # [252000 rows x 13 columns]
    print("loan_train: \n", loan_train)
    # [28000 rows x 12 columns]
    print("loan_test: \n", loan_test)
    print("loan: \n", loan)

    ## Olympic
    # [271116 rows x 15 columns]
    olympic = pd.read_csv("./data/TABULAR/olympic/athlete_events.csv")
    olympic = olympic.drop(columns=["ID", "Name"])
    olympic = olympic.sample(frac=1).reset_index(drop=True)
    process_and_save_data("olympic", olympic, -1, False, r_train=0.7, r_valid=0.1, r_test=0.2)

    print("olympic: \n", olympic)
