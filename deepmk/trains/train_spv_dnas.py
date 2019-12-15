import torch
from deepmk.trains.train_interface import TrainInterface

class TrainSpvDNAS(TrainInterface):
    """
    Supervised training where the update happen in the "training" and "val"
    phases. This is implementation of Differentiable Neural Architecture Search.
    * data: (X, y)
    * y = model(X)
    """
    def __init__(self, **kwargs):
        super(TrainSpvDNAS, self).__init__(**kwargs)

    def update_phase(self, phase, data):
        super(TrainSpvDNAS, self).update_phase(phase, data)

        Xtrue, ytrue = data
        ypred = self.model(Xtrue)
        critval = self.criteria[phase].feed(ytrue, ypred)

        # update the model to minimize the loss function
        if phase == "train" or phase == "val":
            self.opt(phase).zero_grad()
            critval.backward()
            self.opt(phase).step()
            self.update_scheduler(phase)
