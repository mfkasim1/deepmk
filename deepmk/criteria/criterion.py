from abc import ABCMeta, abstractmethod, abstractproperty

class Criterion:
    """
    Criterion objects are to calculate some criteria on batched data.
    The value from the objects is not necessarily derivable.
    """
    @abstractmethod
    def reset(self):
        """
        Reset the criterion calculation and forget all the fed preds and targets
        """
        pass

    @abstractmethod
    def feed(self, preds, targets):
        """
        Feed the object with the prediction and the target.
        Returns the calculated criterion for the given batch.
        """
        pass

    @abstractmethod
    def getval(self):
        """
        Get the calculated criterion for all the batched data.
        """
        pass

    @abstractproperty
    def name(self):
        """
        Return a string of name for this object.
        """
        pass

    @abstractproperty
    def best(self):
        """
        Return "min" if the criterion is best as low as possible, "max" if the
        criterion is best as high as possible, and "unknown" if unknown.
        """
        pass

class MeanCriterion(Criterion):
    def __init__(self, criterion):
        self.criterion = criterion
        self.reset()

    def reset(self):
        self.crit_sum = 0.0
        self.crit_num = 0.0

    def feed(self, preds, targets):
        size = len(targets) * 1.0
        val = self.criterion(preds, targets)
        self.crit_sum += val * size
        self.crit_num += size
        return val

    def getval(self):
        return self.crit_sum / self.crit_num

    @property
    def name(self):
        return "mean " + self.criterion.__class__.__name__

    @property
    def best(self):
        if "loss" in self.name.lower(): return "min"
        return "unknown"
