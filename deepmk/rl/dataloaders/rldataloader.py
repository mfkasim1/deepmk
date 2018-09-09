from abc import ABCMeta, abstractmethod
import numpy as np
from torch.utils.data import DataLoader

__all__ = ["RLDataLoader", "LastStepLoader", "ReplayMemoryLoader"]

class RLDataLoader:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, tuples):
        pass

class LastStepLoader(RLDataLoader):
    def __call__(self, tuples):
        return DataLoader([tuples[-1]])

class ReplayMemoryLoader(RLDataLoader):
    def __init__(self, max_memory=10000, num_batches=1, batch_size=1,
                 shuffle=True, **dataloader_kwargs):
        self.max_memory = max_memory
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataloader_kwargs = dataloader_kwargs

    def __call__(self, tuples):
        tuples = tuples[-self.max_memory:]
        total_size = self.num_batches * self.batch_size
        n = min(len(tuples), total_size)

        # get the element indices to be sampled
        if self.shuffle:
            idx = np.random.choice(range(len(tuples)), size=n, replace=False)
        else:
            idx = range(len(tuples)-n, len(tuples))

        # sample from the tuples and load it to the dataloader
        samples = [tuples[i] for i in idx]
        return DataLoader(samples, batch_size=self.batch_size,
                          **self.dataloader_kwargs)
