import numpy as np
import torch, time
import pickle
from torch.nn import Softmax
import math


def pickle2dict(file_name, key):
    """
    Load dictionary from pickle file
    """
    with open(file_name, "rb") as fIn:
        stored_data = pickle.load(fIn)
        value = stored_data[key]
    return value


def taylor_softmax_v1(x, dim=1, n=2, use_log=False):
    assert n % 2 == 0 and n > 0
    fn = torch.ones_like(x)
    denor = 1.
    for i in range(1, n + 1):
        denor *= i
        fn = fn + x.pow(i) / denor
    out = fn / fn.sum(dim=dim, keepdims=True)
    if use_log: out = out.log()
    return out


class WeightedRandomExplorationStrategy(object):
    """
    Implementation of the Weighted Random Exploration Strategy class defined in the paper :footcite:`killamsetty2023milo`, where we select a set of points based on a global ordering of the dataset.
    Global Ordering has to be provided in prior for selection. We provide a way to compute global ordering for text and image datasets
    using various submodular functions as a util function.
   
    Parameters
    ----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    """

    def __init__(self, trainloader, global_order_file, online=False, temperature=1, per_class=False):
        """
        Constructor method
        """
        self.trainloader = trainloader
        self.N_trn = len(trainloader.sampler.data_source)
        self.online = online
        self.indices = None
        self.gammas = None
        globalorder = pickle2dict(global_order_file, 'globalorder')
        self.global_idxs = np.array([x[0] for x in globalorder])
        self.global_gains = np.array([x[1] for x in globalorder])
        self.global_gains = self.global_gains - self.global_gains.min()
        self.global_gains = np.maximum(self.global_gains, 1e-10)
        self.temperature = temperature 
        self.probs = taylor_softmax_v1(torch.from_numpy(np.array([self.global_gains])/self.temperature)).numpy()[0]
        self.cluster_idxs = pickle2dict(global_order_file, 'cluster_idxs')
        self.per_class = per_class
        self.num_classes = len(list(self.cluster_idxs.keys()))
        #self.probs = softmax(torch.from_numpy(np.array([self.global_gains])/self.temperature)).numpy()[0]

    def select(self, budget):
        """
        Samples subset of size budget from the generated probability distribution.

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
        if self.per_class:
            per_cls_cnt = [len(self.cluster_idxs[key]) for key in self.cluster_idxs.keys()]
            min_cls_cnt = min(per_cls_cnt)
            total_sample_cnt = sum(per_cls_cnt)
            if min_cls_cnt < math.ceil(budget/self.num_classes):
                per_cls_budget = [min_cls_cnt]*self.num_classes
                while sum(per_cls_budget) < budget:
                    for cls in range(self.num_classes):
                        if per_cls_budget[cls] < per_cls_cnt[cls]:
                            per_cls_budget[cls] += 1
            else:
                per_cls_budget = [math.ceil(budget/self.num_classes) for _ in per_cls_cnt]

            
            cluster_labels = list(self.cluster_idxs.keys())
            if self.online:
                self.indices = []
                for i in range(len(cluster_labels)):
                    per_cls_idxs = self.cluster_idxs[cluster_labels[i]]
                    rng = np.random.default_rng(int(time.time()))
                    sel_idxs = rng.choice(per_cls_idxs, size=per_cls_budget[i], replace=False, p=self.probs[per_cls_idxs]/self.probs[per_cls_idxs].sum())
                    sel_idxs = [int(x) for x in sel_idxs]
                    self.indices.extend(sel_idxs)
            elif self.indices is None:
                self.indices = []
                for i in range(len(cluster_labels)):
                    per_cls_idxs = self.cluster_idxs[cluster_labels[i]]
                    sel_idxs = per_cls_idxs[:per_cls_budget[i]]
                    # sel_idxs = [x.item() for x in sel_idxs]
                    self.indices.extend(sel_idxs)
        else:
            if self.online:
                rng = np.random.default_rng(int(time.time()))
                self.indices = rng.choice(self.global_idxs, size=budget, replace=False, p=self.probs)
                self.indices = [int(x) for x in self.indices]
                #self.gammas = torch.ones(budget)
            elif self.indices is None:
                self.indices = self.global_idxs[:budget]
        self.gammas = torch.ones(len(self.indices))
        return self.indices, self.gammas