from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset

class RLAlg:
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self, s, a, r, snext, done):
        """
        Gather the information of the step, update the database, and return
        DataLoader for a training.

        Args:
            s: The initial state.
            a: The action taken.
            r: The reward given in this transition.
            snext: The next state of the transition.
            done: True if the episode ends.

        Returns:
            torch.utils.data.DataLoader or None: DataLoader for the training or
                None. If None, no training will take place. The DataLoader
                should returns tuple of (states, actions, rewards, next_states,
                values).
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
        snext: The next state of the transition.
        done: True if the episode ends.

    Attributes:
        s: The initial state.
        a: The action taken.
        r: The reward given in this transition.
        snext: The next state of the transition.
        done: True if the episode ends.
    """
    def __init__(self, s, a, r, snext, done):
        self.s = s
        self.a = a
        self.r = r
        self.snext = snext
        self.done = done

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
    def __init__(self, list_of_rl_tuple, state_transform=None,
                 target_transform=None):
        self.data = list_of_rl_tuple
        self.state_transform = state_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        tup = self.data[i]
        state = tup.s
        target = tup.get_value()

        # apply transforms
        if self.state_transform is not None:
            state = self.state_transform(state)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (state, tup.a, tup.r, tup.snext, tup.done, \
                target)
