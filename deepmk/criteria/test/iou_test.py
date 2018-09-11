import copy
import unittest
import torch
import numpy as np
from deepmk.criteria import IoU

def getfunc(layer, inp_shape, channel_dim=1, a=0.4, b=0.3, inp_seed=12412):
    def func(cself):
        crit = IoU(last_layer=layer, channel_dim=channel_dim)

        # create a custom target and prediction
        shape = inp_shape
        intersection = 0.0
        union = 0.0
        for i in range(2):
            seed = inp_seed+i
            targets_bin = generate_unique_channel(shape,
                    channel_dim=channel_dim, seed=seed)
            targets_label = targets_bin.argmax(dim=channel_dim)
            preds = generate_unique_channel(shape,
                    channel_dim=channel_dim, a=a, b=b,
                    seed=seed)

            # test with binary labels and the labels
            cself.assertAlmostEqual(crit.feed(preds, targets_bin), 1.0)
            intersection += targets_bin.sum()
            union += targets_bin.sum()

            cself.assertAlmostEqual(crit.feed(preds, 1-targets_bin), 0.0)
            intersection += 0.0
            union += targets_bin.numel()

            cself.assertAlmostEqual(crit.feed(preds, targets_label), 1.0)
            intersection += targets_bin.sum()
            union += targets_bin.sum()

        cself.assertAlmostEqual(crit.getval(), intersection / union)
    return func

class IoUTest(unittest.TestCase):
    def test_sigmoid_2d_wrong_model(self):
        crit = IoU(last_layer="sigmoid")

        # create a custom target and prediction
        shape = (2,3,4,5)
        for i in range(2):
            seed = 13214+i
            targets_bin = generate_unique_channel(shape,channel_dim=1,seed=seed)
            targets_label = targets_bin.argmax(dim=1)
            preds = generate_unique_channel(shape, channel_dim=1, a=0.4, b=0.3,
                    seed=seed)

            # test with binary labels and the labels
            self.assertAlmostEqual(crit.feed(preds, targets_bin), 0.0)
            self.assertAlmostEqual(crit.feed(preds, 1-targets_bin), 0.0)
            self.assertAlmostEqual(crit.feed(preds, targets_label), 0.0)

        self.assertAlmostEqual(crit.getval(), 0.0)

shapes = [(2,3,4), (2,3,4,5), (2,3,4,5,6)]
layers = ["sigmoid", "softmax"]
for shape in shapes:
    for layer in layers:
        a = 0.8 if layer == "sigmoid" else 0.4
        b = 0.3
        for channel_dim in range(1,len(shape)):
            name = "test_%s_%dd_ch%d" % (layer, len(shape)-2, channel_dim)
            func = getfunc(layer, shape, channel_dim, a, b)
            setattr(IoUTest, name, func)

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
