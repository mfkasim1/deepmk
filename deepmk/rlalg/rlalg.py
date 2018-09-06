from abc import ABCMeta, abstractmethod
import torch.utils.data.Dataset as Dataset

class RLAlg:
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self, s, a, r, snext=None):
        """
        Gather the information of the step, update the database, and return
        DataLoader for a training.

        Args:
            s: The initial state.
            a: The action taken.
            r: The reward given in this transition.
            snext: The next state of the transition. None if the episode ends.

        Returns:
            torch.utils.data.DataLoader or None: DataLoader for the training or
                None. If None, no training will take place.
        """
        pass

class RLTuple:
    """
    RLTuple represents a set of information of state transition: state, action,
    reward, and the next state.

    Args:
        s: The initial state.
        a: The action taken.
        r: The reward given in this transition.
        snext: The next state of the transition. None if the episode ends.

    Attributes:
        s: The initial state.
        a: The action taken.
        r: The reward given in this transition.
        snext: The next state of the transition. None if the episode ends.
    """
    def __init__(self, s, a, r, snext):
        self.s = s
        self.a = a
        self.r = r
        self.snext = snext

    def set_value(self, value):
        self.val = value

    def get_value(self):
        return self.val

class RLTupleDataset(Dataset):
    """
    Creates a dataset from a list of RLTuple.
    The __getitem__ method returns a tuple of (state, value).

    Args:
        list_of_rl_tuple (list): List of RLTuple to be formed as a dataset.
    """
    def __init__(self, list_of_rl_tuple):
        self.data = list_of_rl_tuple

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        tup = self.data[i]
        return (tup.s, tup.get_value())
