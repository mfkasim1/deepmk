from abc import abstractmethod, ABCMeta
import deepmk.utils as mkutils
import torch

class TrainInterface(object):
    """
    The parent of Train object.
    This object contains bookkeeping and other menial tasks for training.

    Arguments
    ---------
    * criteria: dict of deepmk.criterion
        The keys are the phases in `dataloader` and the values must
        be functions that produce the criteria to be evaluated against.
    * opt: dict of torch optimizer
        The keys are the phases in `dataloader` and the values are the torch
        optimizer.
    * scheduler: dict of torch scheduler
        The keys are the phases in `dataloader` and the values are the torch
        optimizer.
    * dataloader: deepmk.DataloaderManager
        The object should have "__len__" property and can be iterated.
        In every element, it returns `(phase, data)` where `phase` indicating
        the phase of the training and `data` is any data format to be fed
        on `update_phase` method.
        The phase should have "val" to indicate when to save the model.
    * model: trainable object or list of trainable object
        The model should match the requirement of each Train class.
        See docs in the specific class
    * verbose: bool
        Flag of verbosity (default: 1)
    * numepochs: int
        Number of epochs iterations (default: 1e6)
    * save_crit_to: str or None
        File name to save the criteria values. If None, the criteria is not
        written (default: None)
    * save_wts_to: str or None
        File name to save the model states. If None, the model is not
        saved (default: None)
    """

    def __init__(self, criteria=None, dataloader=None, model=None,
            verbose=1, numepochs=1000000,
            save_crit_to=None, save_wts_to=None):
        self.verbose = verbose
        self.dataloader = dataloader
        self.numepochs = numepochs
        self.save_crit_to = save_crit_to

        self._opt = opt
        self._scheduler = scheduler
        self._model = model
        self._criteria = criteria

        # check the arguments (they cannot be empty)
        self._check_argument(opt, "opt")
        self._check_argument(dataloader, "dataloader")
        self._check_argument(model, "model")
        self._check_argument(criteria, "criteria")

    def _check_argument(self, arg, sname):
        if arg is None:
            raise ValueError("%s argument for Train class cannot be left empty"%\
                             sname)

    def opt(self, phase):
        if type(self._opt) == type({}):
            return self._opt[phase]
        else:
            return self._opt

    def update_scheduler(self, phase):
        if type(self._scheduler) == type({}):
            return self._scheduler[phase].step()
        elif self._scheduler is not None:
            return self._scheduler.step()
        else:
            pass # do nothing

    @property
    def model(self):
        return self._model

    @property
    def criteria(self):
        return self._criteria

    def start(self):
        """
        Method that is called at the very beginning of the training
        """
        # initiate the best criteria
        self.best_critval = None
        self.best_model_state = None

        # write the header of the criteria file
        if self.save_crit_to is not None:
            keys = self.criteria.keys()
            with open(self.save_crit_to, "w") as f:
                s = ",".join(keys)
                f.write(s+"\n")

    def end(self):
        """
        Method that is called at the very end of the training
        """
        pass

    def start_epoch(self):
        """
        Method that is called at the beginning of every epoch
        """
        # reset the criteria
        phases = self.criteria.keys()
        for phase in phases:
            self.criteria[phase].reset()

    def end_epoch(self):
        """
        Method that is called at the end of every epoch
        """
        # save the criteria

        if self.save_crit_to is not None or self.verbose:
            keys = self.criteria.keys()
            vals = ["%.3e"%self.criteria[key].getval() for key in keys]

            if self.save_crit_to is not None:
                s = ",".join(vals)
                with open(self.save_crit_to, "a") as f:
                    f.write(s + "\n")

            if self.verbose:
                ss = ["%s (%s): %.3e" % (keys[i], self.criteria[keys[i]].name, vals[i])]
                s = ", ".join(ss)
                print(s)

        # save the model
        if self.save_wts_to is not None:
            critval = self.criteria["val"].getval()
            save_model = self.best_critval is None or self.best_model_state is None
            save_model = save_model or (self.criteria["val"].best == "max" and critval > self.best_critval)
            save_model = save_model or (self.criteria["val"].best == "min" and critval < self.best_critval)
            if save_model:
                self.best_critval = critval
                self.best_model_state = self._get_model_state()
                torch.save(self.best_model_state)

    @abstractmethod
    def update_phase(self, phase, data):
        """
        Given the phase and the data, update the model
        """
        pass

    def train(self):
        if self.verbose:
            progress_disp = mkutils.ProgressDisplay()

        self.start()

        for i in range(self.numepochs):

            self.start_epoch()

            # display the epoch information
            if self.verbose:
                s = "Epoch %6d/%6d" % (i+1, numepochs)
                print(s)
                print("-" * len(s))

            total_batches = len(dataloader)
            for j,(phase, data) in enumerate(dataloader):

                self.update_phase(phase, data)

                # show the progress bar
                if self.verbose:
                    progress_disp.show(j+1, total_batches)

            self.end_epoch()

            # add more blank spaces
            if self.verbose:
                print("")

        self.end()

    def _get_model_state(self):
        if hasattr(model, "__iter__"):
            return [m.state_dict() for m in model]
        else:
            return model.state_dict()
