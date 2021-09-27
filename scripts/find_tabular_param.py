import pickle

filepaths = ["data/airline.pickle", "data/loan.pickle", "data/olympic.pickle"]

# file: data/airline.pickle, input_dim: 499, n_classes: 2
# file: data/loan.pickle, input_dim: 409, n_classes: 2
# file: data/olympic.pickle, input_dim: 2735, n_classes: 4


if __name__ == "__main__":
    for filepath in filepaths:
        with open(filepath, 'rb') as handle:
            _, _, _, input_dim, n_classes = pickle.load(handle)
        print("file: %s, input_dim: %s, n_classes: %s" % (filepath, input_dim, n_classes))
