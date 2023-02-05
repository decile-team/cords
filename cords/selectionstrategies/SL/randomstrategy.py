import numpy as np
import torch


class RandomStrategy(object):
    """
    This is the Random Selection Strategy class where we select a set of random points as a datasubset
    and often acts as baselines to compare other selection strategies.

    Parameters
    ----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    """

    def __init__(self, trainloader, online=False):
        """
        Constructor method
        """

        self.trainloader = trainloader
        self.N_trn = len(trainloader.sampler.data_source)
        self.online = online
        self.indices = None
        self.gammas = None

    def select(self, budget):
        """
        Perform random sampling of indices of size budget.

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
        if self.online or (self.indices is None):
            np.random.seed()
            self.indices = np.random.choice(self.N_trn, size=budget, replace=False)
            self.gammas = torch.ones(budget)
        self.indices = [int(x) for x in self.indices]
        return self.indices, self.gammas
