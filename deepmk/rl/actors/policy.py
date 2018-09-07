import torch
import numpy as np
from deepmk.rl.actors import Actor

__all__ = ["Policy"]

class Policy(Actor):
    """
    Actor that picks an action, a, stochastically according to the probability
    pi(s,a;t) calculated given a state, s, and the model's parameters, t. It
    maximizes: log(pi(s,a;t))*r(s,a).

    Args:
        model :
            A torch trainable class method that accepts state(s) and returns
            list probability to take each action.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, state):
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
        prob = self.model.forward(state.unsqueeze(dim=0)).squeeze()
        prob = prob / prob.sum() # normalize the probability
        return np.choice(np.linspace(len(prob)), p=prob)

    def value(self, states, actions, rewards, next_states, vals):
        pi = self.model.forward(states).gather(dim=-1, index=actions)
        log_pi = torch.log(pi)
        wlog_pi = log_pi # * ???
        return wlog_pi.sum()
