import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def _create_batch_ordering(self) -> np.ndarray:
        """Creates batch index ordering based on current shuffle setting."""
        indices = (np.random.permutation(self.dataset_size) if self.shuffle 
                  else np.arange(self.dataset_size))
        return np.array_split(
            indices,
            range(self.batch_size, self.dataset_size, self.batch_size)
        )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        # Shuffle data if required
        if self.shuffle:
            self.ordering = self._create_batch_ordering()
            
        # Reset batch counter
        self.current_batch = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        # Check if we've reached the end
        if self.current_batch >= len(self.ordering):
            raise StopIteration
            
        # Get indices for current batch
        batch_indices = self.ordering[self.current_batch]
        
        # Increment batch counter
        self.current_batch += 1
        
        # Load and convert batch data
        batch_samples = self.dataset[batch_indices]
        return [Tensor(sample) for sample in batch_samples]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """Returns the number of batches in the dataloader."""
        return len(self._create_batch_ordering())