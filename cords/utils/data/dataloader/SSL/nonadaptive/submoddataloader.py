import numpy as np
import submodlib
import math
from .nonadaptivedataloader import NonAdaptiveDSSDataLoader


class SubmodDataLoader(NonAdaptiveDSSDataLoader):
    # Currently split dataset with size of |max_chunk| then proportionably select samples in every chunk
    # Otherwise distance matrix will be too large
    """
    Implementation of SubmodDataLoader class for the nonadaptive submodular subset selection strategies for 
    semi-supervised learning setting.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary
    logger: class
        Logger for logging the information
    """
    def __init__(self, train_loader, val_loader, dss_args, logger, *args,
                 **kwargs):
        """
        Constructor function
        """

        super(SubmodDataLoader, self).__init__(train_loader, val_loader, dss_args, 
                                               logger, *args, **kwargs)
        # Arguments assertion check
        assert "size_chunk" in dss_args.keys(), "'size_chunk' is a compulsory agument for submodular dataloader"
        if dss_args.size_chunk:
            self.logger.info("You are using max_chunk: %s" % dss_args.size_chunk)
        self.size_chunk = dss_args.size_chunk
        
    def _init_subset_indices(self):
        """
        Initializes the subset indices and weights by calling the respective submodular function for data subset selection.
        """
        X = np.array([x for (w_x, x, _y) in self.dataset])
        m = X.shape[0]
        # Chunking dataset to calculate pairwise distance with limited memory
        sample_indices = []
        size_chunk, budget = self.size_chunk, self.budget
        n_chunks = math.ceil(m / self.size_chunk)
        budget_chunk = math.ceil(budget / n_chunks)
        for i_chunk in range(n_chunks):
            l_idx = i_chunk * size_chunk
            r_idx = min(m, (i_chunk + 1) * size_chunk)
            n_samples = min(budget_chunk, budget - len(sample_indices))
            chunk = X[l_idx: r_idx, :]
            _sample_indices = self._chunk_select(chunk, n_samples)
            _sample_indices = [_sample_indice + l_idx for _sample_indice in _sample_indices]
            sample_indices += _sample_indices
        return np.array(sample_indices)


# Submodular optimization based

class FacLocDataLoader(SubmodDataLoader):
    """
    Implementation of FacLocDataLoader class for the nonadaptive facility location
    based subset selection strategy for semi-supervised learning setting.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary
    logger: class
        Logger for logging the information
    """
    def _chunk_select(self, chunk, n_samples):
        """
        Function that selects the data samples by calling the facility location function.

        Parameters
        -----------
        chunk: numpy array
            Chunk of the input data from which the subset needs to be selected
        n_samples: int
            Number of samples that needs to be selected from input chunk
        Returns
        --------
        ranking: list
            Ranking of the samples based on the facility location gain 
        """
        fl = submodlib.functions.facilityLocation.FacilityLocationFunction(n=len(chunk), mode='dense', data=chunk)
        return [idx for idx, _ in fl.maximize(n_samples)]


class GraphCutDataLoader(SubmodDataLoader):
    """
    Implementation of GraphCutDataLoader class for the nonadaptive graph cut function
    based subset selection strategy for semi-supervised learning setting.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary
    logger: class
        Logger for logging the information
    """

    def _chunk_select(self, chunk, n_samples):
        """
        Function that selects the data samples by calling the graphcut function.

        Parameters
        -----------
        chunk: numpy array
            Chunk of the input data from which the subset needs to be selected
        n_samples: int
            Number of samples that needs to be selected from input chunk
        Returns
        --------
        ranking: list
            Ranking of the samples based on the graphcut gain 
        """
        fl = submodlib.functions.graphCut.graphCutFunction(n=len(chunk), mode='dense', data=chunk)
        return [idx for idx, _ in fl.maximize(n_samples)]

'''
class SumRedundancyDataLoader(SubmodDataLoader):
    """
    Implementation of SumRedundancyDataLoader class for the nonadaptive sum redundancy function
    based subset selection strategy for semi-supervised learning setting.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary
    logger: class
        Logger for logging the information
    """
    def _chunk_select(self, chunk, n_samples):
        """
        Function that selects the data samples by calling the sum redundancy function.

        Parameters
        -----------
        chunk: numpy array
            Chunk of the input data from which the subset needs to be selected
        n_samples: int
            Number of samples that needs to be selected from input chunk
        Returns
        --------
        ranking: list
            Ranking of the samples based on the sum redundancy gain 
        """
        f = apricot.functions.sumRedundancy.SumRedundancySelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)
'''

class SaturatedCoverageDataLoader(SubmodDataLoader):
    """
    Implementation of SaturatedCoverageDataLoader class for the nonadaptive saturated coverage
    function based subset selection strategy for semi-supervised learning setting.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary
    logger: class
        Logger for logging the information
    """
    def _chunk_select(self, chunk, n_samples):
        """
        Function that selects the data samples by calling the saturated coverage function.

        Parameters
        -----------
        chunk: numpy array
            Chunk of the input data from which the subset needs to be selected
        n_samples: int
            Number of samples that needs to be selected from input chunk
        Returns
        --------
        ranking: list
            Ranking of the samples based on the saturated coverage gain 
        """
        fl = submodlib.functions.saturatedCoverage.saturatedCoverageFunction(n=len(chunk), mode='dense', data=chunk)
        return [idx for idx, _ in fl.maximize(n_samples)]
