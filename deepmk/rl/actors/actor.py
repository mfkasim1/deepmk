from abc import ABCMeta, abstractmethod

__all__ = ["Actor"]

class Actor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, state):
        """
        Returns an action given a state
        """
        pass

    @abstractmethod
    def value(self, states, actions, rewards, next_states, vals):
        """
        Returns a value to be minimized given a set of states, actions, rewards,
        next_states, and the values.

        Args:
            states (torch.tensor) : set of states in form of torch tensor.
            actions (torch.tensor) : set of actions in form of torch tensor.
            rewards (torch.tensor) : set of rewards in form of torch tensor.
            next_states (torch.tensor) : set of next_states in form of torch
                tensor.
            vals (torch.tensor) : set of expected values in form of torch
                tensor.

        Returns:
            torch.tensor : a scalar value to be minimized.
        """
        pass
