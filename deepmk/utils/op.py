import copy
import torch
import numpy as np

def max_to_one(tensor, channel_dim=1):
    """
    Change the maximum value in dim=1 to 1.0 and others to 0.0.
    """
    # get a map of {0,1} with 1 is in place of the maximum in the 1st dimension
    # and others are 0
    _, max_idx = tensor.max(dim=channel_dim)
    return fill_channel(max_idx, tensor.shape[channel_dim],
                        channel_dim=channel_dim)

def fill_channel(tensor, nchannels, channel_dim=1):
    """
    Receives the argmax value in dim=1 of the returned tensor and returns a
    tensor where the value given at the input tensor in dim=1 to be 1.0 and
    others to 0.0.
    """
    device = tensor.device
    tshape = tensor.shape
    res_shape = list(copy.copy(tshape))
    res_shape.insert(channel_dim, nchannels)

    # get the tensor to the right size
    tensor = tensor.unsqueeze(channel_dim)

    # get the correct idx tuple
    tup_idx = []
    for i in range(len(res_shape)):
        if i == channel_dim:
            tup_idx.append(tensor)
        else:
            idx = torch.arange(res_shape[i]).to(device)
            for j in range(len(res_shape)):
                if i == j: continue
                idx = idx.unsqueeze(j)
            tup_idx.append(idx.long())
    tup_idx = tuple(tup_idx)

    # fill 1.0 to the given idx
    res = torch.zeros(*res_shape).to(device)
    res[tup_idx] = 1.0
    return res
