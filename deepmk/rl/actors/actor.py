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
