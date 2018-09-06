from abc import ABCMeta, abstractmethod

class RLActor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, state):
        """
        Returns an action given a state
        """
        pass
