from deepmk.criteria.criterion import Criterion

class Accuracy(Criterion):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def feed(self, preds, targets):
        if len(preds.shape) == 2:
            preds = preds.argmax(dim=1)

        if len(targets.shape) == 2:
            targets = targets.argmax(dim=1)

        correct = (preds == targets)
        num_correct = correct.float().sum()
        size = correct.numel()

        self.correct += num_correct
        self.total += size

        # return the accuracy for this batch
        return num_correct * 1.0 / size

    def getval(self):
        return self.correct * 1.0 / self.total

    @property
    def name(self):
        return "accuracy"

    @property
    def best(self):
        return "max"
