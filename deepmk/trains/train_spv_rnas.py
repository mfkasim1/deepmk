import torch
from deepmk.trains.train_interface import TrainInterface

class TrainSpvRNAS(TrainInterface):
    """
    Supervised training where the update happen in the "training" and "val"
    phases. The update in the "val" phase is REINFORCE update after all the
    epoch has finished.
    The model should return the prediction as well as the logprob of the
    architecture of the model.
    The logprob will be used in the REINFORCE update.
    * data: (X, y)
    * y, logprob = model(X)
    """
    def __init__(self, ranking="hansen", **kwargs):
        super(TrainSpvRNAS, self).__init__(**kwargs)
        self.ranking = ranking

    def start_epoch(self):
        super(TrainSpvRNAS, self).start_epoch()

        # empty the list of logprobs and critvals
        self.logprobs = []
        self.critvals = []

    def update_phase(self, phase, data):
        super(TrainSpvRNAS, self).update_phase(phase, data)

        Xtrue, ytrue = data
        ypred, logprob = self.model(Xtrue)
        critval = self.criteria[phase].feed(ytrue, ypred)

        # update the model to minimize the loss function
        if phase == "train":
            self.opt(phase).zero_grad()
            critval.backward()
            self.opt(phase).step()
            self.update_scheduler(phase)

        elif phase == "val":
            self.logprobs.append(logprob)
            mult = -1.0 if self.criteria[phase].best == "max" else 1.0
            self.critvals.append(critval.data * mult)

    def end_epoch(self):
        super(TrainSpvRNAS, self).end_epoch()

        # perform the REINFORCE update
        normlosses = self._get_normloss(self.critvals)
        loss = (normlosses * self.logprobs).sum()

        # perform the update step
        phase = "val"
        self.opt(phase).zero_grad()
        loss.backward()
        self.opt(phase).step()
        self.update_scheduler(phase)

    def _get_normloss(self, critvals):
        if self.ranking == "hansen":
            """
            Obtain the weight function based on the ranking of the F given by Hansen (tutorial), et al., 2011.
            """
            mu = len(F) * 1.0
            rank = get_rank(F)

            # calculate
            val = np.log(mu + 0.5) - torch.log(rank)
            y = val / torch.sum(val) - (1./mu)
            y = y / y.abs().sum() * mu
            return -y
        else:
            raise ValueError("Undefined ranking %s" % self.ranking)
