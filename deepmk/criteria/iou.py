from deepmk.criteria.criterion import Criterion

class IoU(Criterion):
    """
    Intersect over Union criterion for semantic segmentation. It is fed on
    predictions and targets.
    Predictions have first dimension with the size of the number of object
    classes. The prediction values can be multi-class single label (softmax) or
    multi-class multi-label (sigmoid), but in either case, it has to be [0,1].
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
        ndim0 = preds.shape[0]

        # if the target is presented as a label, then expand it to a binary
        # {0,1} corresponding to the class
        if len(targets.shape) != len(preds.shape):
            targets = _fill_channel(targets, ndim0)

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
    # get a map of {0,1} with 1 is in place of the maximum in the 0-th dimension
    # and others are 0
    _, max_idx = preds.max(dim=0)
    return _fill_channel(max_idx, preds.shape[0])

def _fill_channel(tensor, ndim0):
    numel = tensor.numel()
    map = torch.arange(numel).reshape(tensor.shape)
    idx = tensor * numel + map
    res = torch.zeros(ndim0, *tensor.shape)
    res.view(-1)[idx.view(-1)] = 1.0
    return res
