from torch.utils.data import DataLoader
from deepmk.rl.trainers.trainer import Trainer

__all__ = ["QLearn", "DoubleQLearn"]

class QLearn(Trainer):
    def __init__(self, qnet, optimizer, gamma=0.9, max_memory=1,
                 dataloader_kwargs={}):
        self.qnet = qnet
        self.optimizer = optimizer
        self.tuples = []
        self.gamma = gamma
        self.max_memory = max_memory
        self.dataloader_kwargs = dataloader_kwargs

    def trainstep(self, state, action, reward, next_state, done):
        # save the episode tuple
        tup = (state, action, reward, next_state, done)
        self.tuples.append(tup)
        if len(self.tuples) > self.max_memory: self.tuples.pop(0)

        # set the dataloader
        dataloader = DataLoader(self.tuples, **self.dataloader_kwargs)

        # present the state and the target for the training
        if dataloader is not None:
            for (s, a, r, snext, ep_done) in dataloader:
                # calculate the loss based on the tuple
                snext_val = self.qnet.max_value(snext) * (1.0 - ep_done.float())
                train_target = r.float() + self.gamma * snext_val
                train_state = s.float()
                pred_target = self.qnet.value(s, a)

                loss = ((train_target.data - pred_target)**2).mean()

                # step the optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

class DoubleQLearn(Trainer):
    def __init__(self, qnet1, qnet2, optimizer, gamma=0.9, max_memory=1,
                 dataloader_kwargs={}):
        self.qnet1 = qnet1
        self.qnet2 = qnet2
        self.optimizer = optimizer
        self.tuples = []
        self.gamma = gamma
        self.max_memory = max_memory
        self.dataloader_kwargs = dataloader_kwargs

    def trainstep(self, state, action, reward, next_state, done):
        # save the episode tuple
        tup = (state, action, reward, next_state, done)
        self.tuples.append(tup)
        if len(self.tuples) > self.max_memory: self.tuples.pop(0)

        # set the dataloader
        dataloader = DataLoader(self.tuples, **self.dataloader_kwargs)

        if dataloader is not None:
            for (s, a, r, snext, ep_done) in dataloader:
                snext = snext.float()
                s = s.float()
                r = r.float()
                # calculate the loss based on the tuple
                zero_epdone = (1.0 - ep_done.float())
                qa = self.qnet1.value(s, a)
                qb = self.qnet2.value(s, a)
                qa_next = self.qnet1.value(snext, self.qnet2.max_value(snext,arg=1))*\
                    zero_epdone
                qb_next = self.qnet2.value(snext, self.qnet1.max_value(snext,arg=1))*\
                    zero_epdone
                train_target_a = r + self.gamma * qb_next
                train_target_b = r + self.gamma * qa_next

                loss_a = ((train_target_a.data - qa)**2).mean()
                loss_b = ((train_target_b.data - qb)**2).mean()

                # step the optimizer
                self.optimizer.zero_grad()
                loss_a.backward()
                loss_b.backward()
                self.optimizer.step()
