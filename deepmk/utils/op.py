import torch
import numpy as np

def max_to_one(tensor):
    """
    Change the maximum value in dim=1 to 1.0 and others to 0.0.
    """
    # get a map of {0,1} with 1 is in place of the maximum in the 1st dimension
    # and others are 0
    _, max_idx = tensor.max(dim=1)
    return fill_channel(max_idx, tensor.shape[1])

def fill_channel(tensor, ndim1):
    """
    Receives the argmax value in dim=1 of the returned tensor and returns a
    tensor where the value given at the input tensor in dim=1 to be 1.0 and
    others to 0.0.
    """
    # tensor is in shape (n0, n2, n3) with values indicating which idx in dim=1
    # is to be filled with 1.0
    device = tensor.device
    tshape = tensor.shape
    ndim0 = tensor.shape[0]
    shape = tensor.shape[1:]
    idx2, idx3 = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
        indexing="ij")
    idx2 = torch.from_numpy(idx2).long().to(device) # (n2,n3)
    idx3 = torch.from_numpy(idx3).long().to(device) # (n2,n3)
    idx0 = torch.arange(ndim0).to(device) # (n0,)

    # get the indices right shape
    idx0 = idx0[:,None,None,None]
    tensor = tensor[:,None,:,:]
    idx2 = idx2[None,None,:,:]
    idx3 = idx3[None,None,:,:]

    # map it to the right indices
    res = torch.zeros(ndim0, ndim1, *shape).to(device) # (n0,n1,n2,n3)
    res[idx0,tensor,idx2,idx3] = 1.0
    return res
