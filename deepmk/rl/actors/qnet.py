import numpy as np
import torch
from deepmk.rl.actors import Actor

__all__ = ["QNet"]

class QNet(Actor):
    """
    Actor that picks an action, a, which maximizes the value of Q(s,a;t) given
    a state, s, and the model's parameters, t. Training of the model minimizes
    (Q(s,a;t) - val(s,a))^2.

    Args:
        model :
            A torch trainable class method that accepts state(s) and returns
            list prediction(s) of the next state values.
    """
    def __init__(self, model, gamma=0.9, epsilon=0.1):
        self.model = model
        self.gamma = gamma
        self.epsilon = epsilon

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
        out = self.model.forward(state.unsqueeze(dim=0))
        action_suggest = int(out.argmax(dim=-1)[0])

        # random action with self.epsilon probability
        rand = np.random.random() < self.epsilon
        action = np.random.randint(out.shape[-1]) if rand else action_suggest
        return action

    def value(self, states, actions, rewards, next_states, done, vals):
        # the current state's predicted value
        pred_vals = self.model.forward(states)\
                    .gather(dim=-1, index=actions.unsqueeze(-1))

        # the next state's predicted value
        next_vals = self.model.forward(next_states.float()).max(dim=-1)[0]*self.gamma
        # zeroing out the end of episode
        next_vals = next_vals * (1.-done.float())
        # the target value
        target_vals = next_vals + rewards.float()
        # target_vals = vals.float()
        loss = (target_vals.data - pred_vals)**2
        return loss.mean()
