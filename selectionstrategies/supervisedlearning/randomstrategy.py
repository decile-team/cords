import numpy as np
import torch
from selectionstrategies.supervisedlearning.dataselectionstrategy import DataSelectionStrategy


class RandomStrategy(object):
    """
    This is the Random Selection Strategy class.

    :param trainloader: Loading the training data using pytorch DataLoader
    :type trainloader: class
    :param model: Model architecture used for training
    :type model: class
    :param budget: The number of data points to be selected
    :type budget: int
    :param model_dict: Python dictionary object containing models parameters
    :type model_dict: OrderedDict
    """

    def __init__(self, trainloader, model, budget, model_dict):
        """
        Constructer method
        """

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model
        self.budget = budget
        self.model_dict = model_dict
        self.N_trn = len(trainloader.sampler.data_source)


    def select(self, budget, model_dict):
        """
        Perform random sampling of indices of size budget.

        :param budget: The number of data points to be selected
        :type budget: int
        :param model_dict: Python dictionary object containing models parameters
        :type model_dict: OrderedDict
        :return: Array of indices of size budget selected randomly, and gamma values
        :rtype: ndarray, Tensor
        """

        self.model.load_state_dict(model_dict)
        indxs = np.random.choice(self.N_trn, size=budget, replace=False)
        gammas = torch.ones(budget)
        return indxs, gammas
