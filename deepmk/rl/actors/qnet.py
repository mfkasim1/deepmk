from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from deepmk.rl.actors import Actor

__all__ = ["QNetInterface", "QNet"]

class QNetInterface(Actor):
    """
    QNetInterface interface model to calculate Q(s,a), the value of a given
    state, s, and an action, a. It should also be able to calculate the value
    of a state by taking the value of max_a[Q(s,a)].
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def getaction(self, state):
        pass

    @abstractmethod
    def value(self, states, actions):
        pass

    @abstractmethod
    def max_value(self, states, arg=0):
        pass

class QNet(QNetInterface):
    """
    Actor that picks an action, a, which maximizes the value of Q(s,a;t) given
    a state, s, and the model's parameters, t.

    Args:
        model :
            A torch trainable class method that accepts state(s) and returns
            list prediction(s) of the next state values.
        epsilon (float) :
            Randomly pick an action with probability given by epsilon.
    """
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon

    def getaction(self, state):
        """
        Get the recommended action by taking the action with the largest value
        given by the model.

        Args:
            state :
                The current state.
        Returns:
            int :
                The action in terms of the index.
        """
        state = torch.FloatTensor(state)
        out = self.model.forward(state.unsqueeze(dim=0))
        action_suggest = int(out.argmax(dim=-1)[0])

        # random action with self.epsilon probability
        rand = (np.random.random() < self.epsilon) and self.model.training
        action = np.random.randint(out.shape[-1]) if rand else action_suggest
        return action

    def value(self, states, actions):
        return self.model.forward(states.float())\
            .gather(dim=-1, index=actions.unsqueeze(dim=-1))

    def max_value(self, states, arg=0):
        idx = 1 if arg else 0
        return self.model.forward(states.float()).max(dim=-1)[idx]
