import torch
import numpy as np
from deepmk.criteria.criterion import Criterion

class IoU(Criterion):
    """
    Intersect over Union criterion for semantic segmentation. It is fed on
    predictions and targets.
    Predictions have second dimension (channels) with the size of the number of
    object classes. The prediction values can be multi-class single label
    (softmax) or multi-class multi-label (sigmoid), but in either case, it has
    to be [0,1].
    The targets can be a label, i.e. has the shape similar to predictions (with
    missing the first dimension) with long type elements, or a map of {0,1} with
    the same shape as the predictions.
    """
    def __init__(self, last_layer="sigmoid"):
        self.last_layer = last_layer
        self.reset()

    def reset(self):
        self.intersect = 0.0
        self.union = 0.0

    def feed(self, preds, targets):
        assert len(preds.shape) == 4
        
        ndim1 = preds.shape[1]

        # if the target is presented as a label, then expand it to a binary
        # {0,1} corresponding to the class
        if len(targets.shape) != len(preds.shape):
            targets = _fill_channel(targets, ndim1)

        # change the predictions to have binary values
        if self.last_layer == "softmax":
            preds = _max_to_one(preds)
        elif self.last_layer == "sigmoid":
            preds = (preds > 0.5)

        # convert them to float
        preds = preds.float()
        targets = targets.float()

        # calculate the intersects and the unions
        intersect = (preds * targets).sum()
        union = ((preds + targets) > 0.0).float().sum()

        self.intersect += intersect
        self.union += union

        # return the accuracy for this batch
        return intersect * 1.0 / union

    def getval(self):
        return self.intersect * 1.0 / self.union

    @property
    def name(self):
        return "IoU"

    @property
    def best(self):
        return "max"

def _max_to_one(tensor):
    # get a map of {0,1} with 1 is in place of the maximum in the 1st dimension
    # and others are 0
    _, max_idx = tensor.max(dim=1)
    return _fill_channel(max_idx, tensor.shape[1])

def _fill_channel(tensor, ndim1):
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
