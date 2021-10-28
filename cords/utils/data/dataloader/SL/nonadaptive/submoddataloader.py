import numpy as np
import apricot
import math
from .nonadaptivedataloader import NonAdaptiveDSSDataLoader
import torch


class SubmodDataLoader(NonAdaptiveDSSDataLoader):
    # Currently split dataset with size of |max_chunk| then proportionably select samples in every chunk
    # Otherwise distance matrix will be too large
    def __init__(self, train_loader, val_loader, dss_args, logger, *args,
                 **kwargs):
        assert "size_chunk" in dss_args.keys(), "'size_chunk' is a compulsory agument for submodular dataloader"
        self.size_chunk = dss_args.size_chunk
        self.dss_args = dss_args
        super(SubmodDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                               logger, *args, **kwargs)
        self.logger.info("You are using max_chunk: %s", dss_args.size_chunk) 

    def _init_subset_indices(self): 
        for i, (x, y) in enumerate(self.train_loader):
            if i == 0:
                if self.dss_args.data_type == 'text':
                    with torch.no_grad():
                        X = self.dss_args.model.embedding(x.to(self.device))
                    X = X.mean(dim = 1)
                    X = X.reshape(X.shape[0], -1)
                else:
                    X = x
                    X = X.reshape(X.shape[0], -1)
            else:
                if self.dss_args.data_type == 'text':
                    with torch.no_grad():
                        X_b = self.dss_args.model.embedding(x.to(self.device))
                    X_b = X_b.mean(dim = 1)
                    X_b = X_b.reshape(X_b.shape[0], -1)
                else:
                    X_b = x
                    X_b = X_b.reshape(X_b.shape[0], -1)
                X = torch.cat((X, X_b), dim=0)
        m = X.shape[0]
        X = X.to(device='cpu').numpy()
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
    def __init__(self, train_loader, val_loader, dss_args, logger, *args,
                 **kwargs):
        
        super(FacLocDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                               logger, *args, **kwargs)

    def _chunk_select(self, chunk, n_samples):
        f = apricot.functions.facilityLocation.FacilityLocationSelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)


class GraphCutDataLoader(SubmodDataLoader):

    def __init__(self, train_loader, val_loader, dss_args, logger, *args,
                 **kwargs):
        
        super(GraphCutDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                               logger, *args, **kwargs)

    def _chunk_select(self, chunk, n_samples):
        f = apricot.functions.graphCut.GraphCutSelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)


class SumRedundancyDataLoader(SubmodDataLoader):

    def __init__(self, train_loader, val_loader, dss_args, logger, *args,
                 **kwargs):
        
        super(SumRedundancyDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                               logger, *args, **kwargs)

    def _chunk_select(self, chunk, n_samples):
        f = apricot.functions.sumRedundancy.SumRedundancySelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)


class SaturatedCoverageDataLoader(SubmodDataLoader):

    def __init__(self, train_loader, val_loader, dss_args, logger, *args,
                 **kwargs):
        
        super(SaturatedCoverageDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                               logger, *args, **kwargs)

    def _chunk_select(self, chunk, n_samples):
        f = apricot.functions.facilityLocation.FacilityLocationSelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)
