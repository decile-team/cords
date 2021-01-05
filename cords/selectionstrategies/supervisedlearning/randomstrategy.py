import numpy as np
import torch


class RandomStrategy(object):
    """
    This is the Random Selection Strategy class.
    
    :param trainloader: Loading the training data using pytorch DataLoader
    :type trainloader: class
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
        
        :param budget: The number of data points to be selected
        :type budget: int
        :return: Array of indices of size budget selected randomly, and gamma values
        :rtype: ndarray, Tensor
        """

        indxs = np.random.choice(self.N_trn, size=budget, replace=False)
        gammas = torch.ones(budget)
        return indxs, gammas