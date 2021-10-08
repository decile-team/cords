from typing import TypeVar, Sequence
from torch.utils.data import Dataset

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class WeightedSubset(Dataset[T_co]):
    r"""
    Subset of a dataset with weights at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        weights (sequence): Weights of the subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]
    weights: Sequence[float]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], weights: Sequence[float]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        tmp_list = list(self.dataset[self.indices[idx]])
        tmp_list.append(self.weights[idx])
        return tuple(tmp_list)

    def __len__(self):
        return len(self.indices)