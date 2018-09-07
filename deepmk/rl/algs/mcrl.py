import torch.utils.data.DataLoader as DataLoader
from deepmk.rl.algs.rlalg import RLAlg, RLTuple, RLTupleDataset

__all__ = ["MonteCarloRL"]

class MonteCarloRL(RLAlg):
    """
    MonteCarlo reinforcement learning: only do the training after each an
    episode with the value of a state is a reward accummulation from the end
    of the episode.

    Args:
        dataloader_kwargs (dict): The keyword arguments for
            torch.utils.data.DataLoader (default: {})
    """
    def __init__(self, dataloader_kwargs={}):
        self.episode = []

    def step(self, s, a, r, snext=None):
        # save the tuple
        self.episode.append(RLTuple(s, a, r, snext))

        # if not the end of an episode, then return None (i.e. no training)
        if snext is not None: return None

        # end of an episode, calculate the value of each state
        value = 0
        for i in range(len(self.episode)-1,-1,-1):
            tup = self.episode[i]
            value += tup.r # increase the collective value
            tup.set_value(value) # save the value of the state

        # construct a dataloader to be returned
        # the dataset should contains all the transitions in the episode
        dataset = RLTupleDataset(self.episode)
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        return dataloader