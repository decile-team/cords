import numpy as np
import apricot
import math
from .nonadaptivedataloader import NonAdaptiveDSSDataLoader


class SubmodDataLoader(NonAdaptiveDSSDataLoader):
    # Currently split dataset with size of |max_chunk| then proportionably select samples in every chunk
    # Otherwise distance matrix will be too large
    def __init__(self, train_loader, val_loader, dss_args, logger, *args,
                 **kwargs):
        
        super(SubmodDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                               logger, *args, **kwargs)

        assert "size_chunk" in dss_args.keys(), "'size_chunk' is a compulsory agument for submodular dataloader"
        if dss_args.size_chunk:
            self.logger.info("You are using max_chunk: %s" % dss_args.size_chunk)
        self.size_chunk = dss_args.size_chunk
        
    def _init_subset_indices(self):
        X = np.array([x for (x, _y) in self.dataset])
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
    def _chunk_select(self, chunk, n_samples):
        f = apricot.functions.facilityLocation.FacilityLocationSelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)


class GraphCutDataLoader(SubmodDataLoader):
    def _chunk_select(self, chunk, n_samples):
        f = apricot.functions.graphCut.GraphCutSelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)


class SumRedundancyDataLoader(SubmodDataLoader):
    def _chunk_select(self, chunk, n_samples):
        f = apricot.functions.sumRedundancy.SumRedundancySelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)


class SaturatedCoverageDataLoader(SubmodDataLoader):
    def _chunk_select(self, chunk, n_samples):
        f = apricot.functions.facilityLocation.FacilityLocationSelection(n_samples=n_samples)
        m = f.fit(chunk)
        return list(m.ranking)
