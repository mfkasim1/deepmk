from deepmk.criteria.criterion import Criterion

class IoU(Criterion):
    """
    Intersect over Union criterion for semantic segmentation. It is fed on
    predictions and targets. Predictions have last dimension with the size of
    the number of object classes and the values are the confidence on the object
    (0-1). Targets have the same shape as predictions with values {0,1}.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.intersect = 0.0
        self.union = 0.0

    def feed(self, preds, targets):
        preds = (preds > 0.5).float()
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
