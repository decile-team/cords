import numpy as np
import torch
import pickle
import random


def pickle2dict(file_name, key):
    """
    Load dictionary from pickle file
    """
    with open(file_name, "rb") as fIn:
        stored_data = pickle.load(fIn)
        value = stored_data[key]
    return value


class StochasticGreedyExplorationStrategy(object):
    """
    This is the Stochastic Greedy Exploration Strategy class defined in the paper :footcite:`killamsetty2023milo`, where we select multiple subsets using stochastic greedy algorithm.
    Stochastic subsets has to be provided in prior for selection. We provide a way to compute stochastic subsets for text and image datasets
    using various submodular functions as a util function.
   
    Parameters
    ----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    """

    def __init__(self, trainloader, stochastic_subsets_file):
        """
        Constructor method
        """
        self.trainloader = trainloader
        self.N_trn = len(trainloader.sampler.data_source)
        self.indices = None
        self.gammas = None
        stochasticsubsets = pickle2dict(stochastic_subsets_file, 'stochastic_subsets')
        self.stochastic_subsets = []
        for subset in stochasticsubsets:
            self.stochastic_subsets.append([x[0] for x in subset])
        random.shuffle(self.stochastic_subsets)
        self.sel_idx = 0
        

    def select(self, budget):
        """
        Utilizes already selected stochastic subsets of indices of size budget.

        Parameters
        ----------
        budget: int
            The number of data points to be selected

        Returns
        ----------
        indices: ndarray
            Array of indices of size budget selected randomly
        gammas: Tensor
            Gradient weight values of selected indices
        """
        self.indices = self.stochastic_subsets[self.sel_idx]
        self.indices = [int(x) for x in self.indices]
        self.sel_idx  = (self.sel_idx+1) % len(self.stochastic_subsets)
        self.gammas = torch.ones(len(self.indices))
        return self.indices, self.gammas