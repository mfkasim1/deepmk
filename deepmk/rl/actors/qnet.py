import torch
from deepmk.rl.actors import Actor

"""
Actors with a single model.
"""

__all__ = ["QNet", "PolicyActor"]

class QNet(Actor):
    """
    Go to the state that has the largest value of Q(s,a) given by the model.
    The model must receive an input of a state and output the values of the
    next state per action.

    Args:
        model :
            A torch trainable class method that accepts state(s) and returns
            list prediction(s) of the next state values.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, state):
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
        out = self.model.forward(state.unsqueeze(dim=0)).argmax(dim=-1)[0]
        return int(out)

    def value(self, states, actions, rewards, next_states, vals):
        pred_vals = self.model.forward(states).gather(dim=-1, index=actions)
        loss = (vals - pred_vals).square()
        return loss
