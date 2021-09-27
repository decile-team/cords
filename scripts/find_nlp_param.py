import pickle

# filepaths = ["data/corona.pickle", "data/news.pickle", "data/twitter.pickle"]
filepaths = ["data/corona.pickle", "data/news.pickle", "data/twitter.pickle", "data/ag_news.pickle"]

# file: data/corona.pickle, vocab_size: 104590, n_classes: 5
# file: data/news.pickle, vocab_size: 176230, n_classes: 2
# file: data/twitter.pickle, vocab_size: 50233, n_classes: 4

if __name__ == "__main__":
    for filepath in filepaths:
        with open(filepath, 'rb') as handle:
            _, _, _, vocab_size, n_classes = pickle.load(handle)
        print("file: %s, vocab_size: %s, n_classes: %s" % (filepath, vocab_size, n_classes))
