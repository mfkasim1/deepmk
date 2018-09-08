from abc import ABCMeta, abstractmethod

__all__ = ["Trainer"]

class Trainer:
    @abstractmethod
    def trainstep(self, state, action, reward, next_state, done):
        """
        Doing a training step after each step/episode.
        """
        pass
