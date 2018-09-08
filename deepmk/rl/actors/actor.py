from abc import ABCMeta, abstractmethod

__all__ = ["Actor"]

class Actor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def getaction(self, state):
        """
        Get the recommended action by taking the action randomly according to
        the probability given by the model.

        Args:
            state :
                The current state.
        Returns:
            int :
                The action in terms of the index.
        """
        pass
