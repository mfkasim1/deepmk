from collections import deque
from torch.utils.data import DataLoader
from deepmk.rl.trainers.trainer import Trainer
from deepmk.rl.dataloaders import LastStepLoader

__all__ = ["QLearn"]

class QLearn(Trainer):
    def __init__(self, qnet, optimizer, gamma=0.9,
                 rldataloader=LastStepLoader()):
        self.qnet = qnet
        self.optimizer = optimizer
        self.tuples = []
        self.gamma = gamma
        self.rldataloader = rldataloader

    def trainstep(self, state, action, reward, next_state, done):
        # save the episode tuple
        tup = (state, action, reward, next_state, done)
        self.tuples.append(tup)

        # set the dataloader
        dataloader = self.rldataloader(self.tuples)

        # present the state and the target for the training
        if dataloader is not None:
            for s, a, r, snext, ep_done in dataloader:
                # calculate the loss based on the tuple
                snext_val = self.qnet.max_value(snext) * (1.0 - ep_done.float())
                train_target = r.float() + self.gamma * snext_val
                pred_target = self.qnet.value(s, a)

                loss = ((train_target.data - pred_target)**2).mean()

                # step the optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
