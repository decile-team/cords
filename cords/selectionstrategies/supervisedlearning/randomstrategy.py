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

    def __init__(self, trainloader):
        """
        Constructer method
        """

        self.trainloader = trainloader
        self.N_trn = len(trainloader.sampler.data_source)


    def select(self, budget):
        """
        Perform random sampling of indices of size budget.
        
        Parameters
        ----------
        budget: int
            The number of data points to be selected
        
        Returns
        ----------
        indxs: ndarray
            Array of indices of size budget selected randomly
        gammas: Tensor
            Gradient values of selected indices
        """

        indxs = np.random.choice(self.N_trn, size=budget, replace=False)
        gammas = torch.ones(budget)
        return indxs, gammas