import torch
import numpy as np
from deepmk.criteria.criterion import Criterion
import deepmk.utils as mkutils

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
            targets = mkutils.fill_channel(targets, ndim1)

        # change the predictions to have binary values
        if self.last_layer == "softmax":
            preds = mkutils.max_to_one(preds)
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
