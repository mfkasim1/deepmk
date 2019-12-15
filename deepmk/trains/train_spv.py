import torch
from deepmk.trains.train_interface import TrainInterface

class TrainSpv(TrainInterface):
    """
    Supervised training where the update only happen in "training" phase.
    * data: (X, y)
    * y = model(X)
    """
    def __init__(self, **kwargs):
        super(TrainSpv, self).__init__(**kwargs)

    def update_phase(self, phase, data):
        super(TrainSpv, self).update_phase(phase, data)

        Xtrue, ytrue = data
        ypred = self.model(Xtrue)
        critval = self.criteria[phase].feed(ytrue, ypred)

        # update the model to minimize the loss function
        if phase == "train":
            self.opt(phase).zero_grad()
            critval.backward()
            self.opt(phase).step()
            self.update_scheduler(phase)
