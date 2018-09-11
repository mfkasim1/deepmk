import copy
import unittest
import torch
import numpy as np
from deepmk.criteria import IoU

class IoUTest(unittest.TestCase):
    def test_softmax_2d(self):
        crit = IoU(last_layer="softmax")

        # create a custom target and prediction
        shape = (2,3,4,5)
        intersection = 0.0
        union = 0.0
        for i in range(2):
            seed = 13214+i
            targets_bin = generate_unique_channel(shape,channel_dim=1,seed=seed)
            targets_label = targets_bin.argmax(dim=1)
            preds = generate_unique_channel(shape, channel_dim=1, a=0.8, b=0.1,
                    seed=seed)

            # test with binary labels and the labels
            self.assertAlmostEqual(crit.feed(preds, targets_bin), 1.0)
            intersection += targets_bin.sum()
            union += targets_bin.sum()

            self.assertAlmostEqual(crit.feed(preds, 1-targets_bin), 0.0)
            intersection += 0.0
            union += targets_bin.numel()

            self.assertAlmostEqual(crit.feed(preds, targets_label), 1.0)
            intersection += targets_bin.sum()
            union += targets_bin.sum()

        self.assertAlmostEqual(crit.getval(), intersection / union)

def generate_unique_channel(shape, channel_dim=1, a=1.0, b=0.0, seed=None):
    if seed is not None: np.random.seed(seed)

    ndim = len(shape)
    assert channel_dim < ndim

    other_shape = list(copy.copy(shape))
    nchannels = other_shape.pop(channel_dim)

    # generate the index for the random channels
    rand_channels = np.random.randint(low=0, high=nchannels, size=other_shape)
    rand_channels = torch.from_numpy(rand_channels).unsqueeze(channel_dim)

    # get the tuple for the indices
    tup_idx = []
    for i in range(ndim):
        if i == channel_dim:
            tup_idx.append(rand_channels.long())
        else:
            idx = torch.arange(shape[i])
            for j in range(ndim):
                if i == j: continue
                idx = idx.unsqueeze(j)
            tup_idx.append(idx.long())

    tup_idx = tuple(tup_idx)

    res = torch.zeros(shape) + b
    res[tup_idx] = a
    return res
