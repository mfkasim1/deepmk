import time
import copy
import shutil
import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import deepmk.utils as mkutils
import deepmk.criteria

"""
This file contains method to train supervised learning model and
optimize the validation data by REINFORCE.
"""

__all__ = ["train"]

def train(model, dataloaders, criteria, optimizer, scheduler=None,
          num_epochs=25, device=None, verbose=1, plot=0, save_wts_to=None,
          save_model_to=None, return_history=False, return_best_last=9e99,
          memory_size=0, memory_step=0):
    """
    Performs a training of the model.

    Args:
        model :
            A torch trainable class method that accepts "inputs" and returns
            prediction of "outputs".
            The model needs to return the output and the logprobability.
        dataloaders (dict or torch.utils.data.DataLoader):
            Dictionary with two keys: ["train", "val"] with every value is an
            iterable with two outputs: (1) the "inputs" to the model and (2) the
            ground truth of the "outputs". If it is a DataLoader, then it's only
            for the training, nothing for validation.
        criteria (dict or callable or deepmk.criteria):
            Dictionary with two keys: ["train", "val"] with every value is a
            callable or deepmk.criteria to calculate the criterion for the
            corresponding phase. If it is not a dictionary, then the criterion
            is set for both training and validation phases.
            If it is a callable, it is wrapped by deepmk.criteria.MeanCriterion
            object to calculate the mean criterion.
            The criterion for the training needs to be differentiable and it
            will be minimized during the training.
        optimizer (torch.optim optimizer or dict):
            Optimizer class in training the model. If it is a dictionary, it
            must have "train" and "val" keys and it makes it a meta-learning
            problem.
        scheduler (torch.optim.lr_scheduler object or dict):
            Scheduler of how the learning rate is evolving through the epochs. If it
            is None, it does not update the learning rate. It can be a dictionary
            like the optimizer argument. (default: None)
        num_epochs (int):
            The number of epochs in training. (default: 25)
        device :
            Device where to do the training. None to choose cuda:0 if available,
            otherwise, cpu. (default: None)
        verbose (int):
            The level of verbosity from 0 to 1. (default: 1)
        plot (int):
            Whether to plot the loss of training and validation data. (default: 0)
        save_wts_to (str):
            Name of a file to save the best model's weights. If None, then do not
            save. (default: None)
        save_model_to (str):
            Name of a file to save the best whole model. If None, then do not save.
            (default: None)
        return_history (bool):
            A flag to indicate whether the training and validation losses history
            will be returned. (default: False)
        return_best_last (int):
            Return the best model over the last `return_best_last` epochs.
            (default: 9e99)
        memory_size (int):
            Size of the replay memory. (default: 0)
        memory_step (int):
            The number of learning steps to be taken from the replay memory.
            (default: 0)

    Returns:
        best_model :
            The trained model with the lowest loss criterion during "val" phase
    """
    # get the device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:")
    print(device)

    # check some variables if they are dictionary with "train" and "val" keys
    def _check(var, name):
        if var is None: return
        if not (type(var) == dict and "train" in var and "val" in var):
            raise TypeError("The variable %s must be a dictionary with "
                            "'train' and 'val' in it")
    _check(optimizer, "optimizer")
    _check(scheduler, "scheduler")
    _check(dataloaders, "dataloaders")
    _check(criteria, "criteria")

    # set interactive plot
    if plot:
        plt.ion()

    for phase in ["train", "val"]:
        if not issubclass(criteria[phase].__class__, deepmk.criteria.Criterion):
            criteria[phase] = deepmk.criteria.MeanCriterion(criteria[phase])
        criteria[phase].reset()

    # prepare the memory of the last best weights
    if return_best_last < num_epochs:
        weights_history = [None for _ in range(return_best_last)]

    # load the model to the device first
    model = model.to(device)

    if verbose >= 1:
        since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    train_losses = []
    val_losses = []

    total_batches = len(dataloaders["train"]) + len(dataloaders["val"])

    # set up the replay memory
    if memory_size > 0:
        rmem_losses = torch.zeros(memory_size).to(device)
        rmem_logps = torch.zeros(memory_size).to(device)
        memidx = 0
        memfilled = 0
    try:
        best_epoch = 0
        for epoch in range(num_epochs):
            if verbose >= 1:
                print("Epoch %d/%d" % (epoch+1, num_epochs))
                print("-"*10)

            # to time the progress
            epoch_start_time = time.time()

            # progress counter
            num_batches = 0 # num batches in training and validation
            if verbose >= 2:
                progress_disp = mkutils.ProgressDisplay()

            # to store the losses in validation for REINFORCE
            losses = torch.zeros(len(dataloaders["val"])).to(device)
            logps = torch.zeros(len(dataloaders["val"])).to(device)

            # every epoch has a training and a validation phase
            for phase in ["train", "val"]:

                # skip phase if the dataloaders for the current phase is empty
                if dataloaders[phase] == []: continue

                # set the model's mode
                if scheduler is not None:
                    scheduler[phase].step()
                model.train()

                # the total loss during this epoch
                running_loss = 0.0

                # iterate over the data
                dataset_size = 0

                # reset the criteria before the training epoch starts
                criteria[phase].reset()
                count_i = 0
                for inputs, labels in dataloaders[phase]:
                    # get the size of the dataset
                    dataset_size += inputs.size(0)
                    num_batches += 1

                    # write the progress bar
                    if verbose >= 2:
                        progress_disp.show(num_batches, total_batches)

                    # load the inputs and the labels to the working device
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # reset the model gradient to 0
                    optimizer["train"].zero_grad()
                    optimizer["val"].zero_grad()

                    # forward
                    outputs, logp = model(inputs)
                    loss = criteria[phase].feed(outputs, labels)

                    # backward gradient computation and optimize in training
                    if phase == "train":
                        loss.backward()
                        optimizer[phase].step()
                    else:
                        # we need the gradient for logp, but not for loss
                        losses[count_i] += loss.data
                        logps[count_i] += logp
                    count_i += 1

                # do the reinforce
                if phase == "val":
                    # transform the loss into some ranking function (min loss lower)
                    normlosses = get_normloss(losses)
                    # we choose sum instead of mean because the training step
                    # is only done once, so we want to make it larger
                    # (it is approximately mean, but doing it for every batch)
                    loss = (normlosses * logps).sum()
                    rg = memory_step > 0 and memory_size > 0
                    loss.backward(retain_graph=rg)
                    optimizer[phase].step()

                    # pick some from replay memory randomly
                    if rg:
                        num_to_pick = len(losses)
                        num_memory = np.minimum(memory_size, memfilled)
                        print(num_memory)
                        memstep = np.minimum(memory_step, num_memory // num_to_pick)
                        for _ in range(memstep):
                            # get the index from the replay memory to pick
                            mem_probs = torch.zeros(num_memory) + 1./num_memory
                            idx_pick = torch.multinomial(mem_probs, num_to_pick)

                            # calculate the loss function from the replay memory
                            rm_losses = rmem_losses[idx_pick]
                            rm_logps = rmem_logps[idx_pick]
                            norm_losses = get_normloss(rm_losses)
                            rm_loss = (norm_losses * rm_logps).sum()

                            # do the step
                            optimizer[phase].zero_grad()
                            rm_loss.backward(retain_graph=True)
                            optimizer[phase].step()

                        # save the new losses and logps to the replay memory
                        if memidx + len(losses) <= memory_size:
                            rmem_losses[memidx:memidx+len(losses)] = losses
                            rmem_logps[memidx:memidx+len(losses)] = logps
                            memidx += len(losses)
                        elif memory_size > 0:
                            remsize = memory_size - memidx
                            rem2size = len(losses) - remsize
                            rmem_losses[memidx:] = losses[:remsize]
                            rmem_logps[memidx:] = logps[:remsize]
                            rmem_losses[:rem2size] = losses[remsize:]
                            rmem_logps[:rem2size] = logps[remsize:]
                            memidx = rem2size
                        memfilled += len(losses)

                # get the mean loss in this epoch
                mult = -1 if (criteria[phase].best == "max") else 1
                crit_val = criteria[phase].getval()
                epoch_loss = mult * crit_val

                # save the losses
                if phase == "train":
                    train_losses.append(crit_val.data)
                elif phase == "val":
                    val_losses.append(crit_val.data)

                # save the model history
                if return_best_last < num_epochs:
                    weights_history[epoch % return_best_last] = copy.deepcopy(model.state_dict())

                # copy the best model
                if phase == "val" and \
                        ((epoch_loss < best_loss) or \
                         (epoch - best_epoch > return_best_last)):
                    if epoch - best_epoch > return_best_last:
                        # get the index of the next best last
                        val_losses_n = val_losses[-return_best_last:]
                        min_idx_rel = np.argmin(val_losses_n)
                        min_idx = min_idx_rel + len(val_losses) - return_best_last

                        # get the best conditions
                        best_epoch = min_idx
                    else:
                        best_epoch = epoch

                    # save the best conditions
                    best_loss = val_losses[best_epoch]
                    best_model_weights = weights_history[best_epoch % return_best_last]

                    # save the model
                    _save_wts(best_model_weights, save_wts_to)

            # show the loss in the current epoch
            if verbose >= 1:
                print("train %s: %.4e, val %s: %.4e, done in %fs (best val: %.3e)" % \
                      (criteria["train"].name, train_losses[-1],
                       criteria["val"].name, val_losses[-1],
                       time.time()-since, best_loss))
            # plot the losses
            if plot:
                xs_plot = range(1,epoch+2)
                plt.clf()
                plt.plot(xs_plot, train_losses, 'o-')
                plt.plot(xs_plot, val_losses, 'o-')
                plt.legend(["Train", "Validation"])
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.pause(0.001)

            print("")
    except KeyboardInterrupt:
        print("Interrupted. Returning the results.")

    if verbose >= 1:
        time_elapsed = time.time()- since
        print("Training complete in %fs" % time_elapsed)
        print("Best val loss: %.4f" % best_loss)

    # return the model
    model.load_state_dict(best_model_weights)
    if return_history:
        return model, best_loss, train_losses, val_losses
    return model, best_loss

def get_normloss(F):
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

def get_rank(F):
    # number of elements in float
    lenF = len(F)
    # get the ranking from 1..lmbda where 1 is the best (lowest)
    idx_sort = torch.argsort(F).to(F.device)
    rank = torch.zeros(lenF).type(F.type()).to(F.device)
    rank[idx_sort] = torch.arange(lenF).type(F.type()).to(F.device)
    rank = rank + 1
    return rank

def _save_wts(wts, save_wts_to):
    if save_wts_to is not None:
        mkutils.save(wts, save_wts_to)
