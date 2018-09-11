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
    def __init__(self, last_layer="sigmoid", channel_dim=1,
                 exclude_channels=None):
        self.last_layer = last_layer
        self.channel_dim = channel_dim
        self.exclude_channels = self._normalize_exclude_channels(
                                exclude_channels)
        self.reset()

    def reset(self):
        self.intersect = 0.0
        self.union = 0.0

    def feed(self, preds, targets):
        assert len(preds.shape) > 2

        ndim1 = preds.shape[self.channel_dim]

        # if the target is presented as a label, then expand it to a binary
        # {0,1} corresponding to the class
        if len(targets.shape) != len(preds.shape):
            targets = mkutils.fill_channel(targets, ndim1,
                                           channel_dim=self.channel_dim)

        # change the predictions to have binary values
        if self.last_layer == "softmax":
            preds = mkutils.max_to_one(preds, channel_dim=self.channel_dim)
        elif self.last_layer == "sigmoid":
            preds = (preds > 0.5)

        # convert them to float
        preds = preds.float()
        targets = targets.float()

        # exclude some channels, if specified
        if self.exclude_channels is not None:
            preds, targets = self._exclude(preds, targets)

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

    def _normalize_exclude_channels(self, exclude_channels):
        # preprocess exclude_channels
        if exclude_channels is not None:
            if not hasattr(exclude_channels, "__iter__"):
                exclude_channels = [exclude_channels]
        return exclude_channels

    def _exclude(self, preds, targets):
        # exclude the channels in preds and targets as specified in
        # self.exclude_channels
        nchannels = preds.shape[self.channel_dim]
        channels = np.arange(nchannels)
        channels = np.delete(channels, self.exclude_channels)
        channels = torch.from_numpy(channels).long().to(preds.device)

        # construct the tuple idx to only include channels in `channels`
        tup_idx = []
        for i in range(len(preds.shape)):
            if i == self.channel_dim:
                tup_idx.append(channels)
            else:
                tup_idx.append(slice(None,None,None))
        tup_idx = tuple(tup_idx)

        # apply the indexing
        preds = preds[tup_idx] # [:,channels,:,:]
        targets = targets[tup_idx] # [:,channels,:,:]
        return preds, targets
