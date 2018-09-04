import torch
import numpy as np

__all__ = ["RandomCropTensor"]

class RandomCropTensor(object):
    """
    Operate the random crop on 2D image on torch tensor format with size
    (c x h x w). This is similar to torchvision.transforms.RandomCrop
    """
    def __init__(self, size, pad_if_needed=False):
        self.size = size if hasattr(size, '__iter__') else (size, size)
        self.pad_if_needed = pad_if_needed

    def __call__(self, x):
        max_idx0 = x.shape[1] - self.size[0]
        max_idx1 = x.shape[2] - self.size[1]

        # pad if needed
        if self.pad_if_needed and max_idx0 < 0:
            pad_shape = [x.shape[0], -max_idx0, x.shape[2]]
            x = torch.cat((x, torch.zeros(*pad_shape)), dim=1)
            max_idx0 = 0
        if self.pad_if_needed and max_idx1 < 0:
            pad_shape = [x.shape[0], x.shape[1], -max_idx1]
            x = torch.cat((x, torch.zeros(*pad_shape)), dim=2)
            max_idx1 = 0

        # get the initial crop index
        idx0 = np.random.randint(max_idx0+1)
        idx1 = np.random.randint(max_idx1+1)

        return x[:,idx0:idx0+self.size[0],idx1:idx1+self.size[1]]
