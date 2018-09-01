import time
import copy
import shutil
import os
import torch
import torch.nn as nn
import torchvision
import numpy as np

"""
This file contains method to train and test supervised learning model.
"""

__all__ = ["train"]

def train(model, dataloaders, criterion, optimizer, scheduler=None,
          num_epochs=25, device=None, verbose=1, save_wts_to=None,
          save_model_to=None):
    """
    Performs a training of the model.

    Parameters
    ----------
    model :
        A torch trainable class method that accepts "inputs" and returns
        prediction of "outputs".
    dataloaders : dict
        Dictionary with two keys: ["train", "val"] with every value is an
        iterable with two outputs: (1) the "inputs" to the model and (2) the
        ground truth of the "outputs".
    criterion : function or evaluable class
        Receives the prediction of the "outputs" as the first argument, and
        the ground truth of the "outputs" as the second argument. It returns
        the loss function to be minimized.
    optimizer : torch.optim optimizer
        Optimizer class in training the model.
    scheduler : torch.optim.lr_scheduler object
        Scheduler of how the learning rate is evolving through the epochs. If it
        is None, it does not update the learning rate. (default: None)
    num_epochs : int
        The number of epochs in training. (default: 25)
    device :
        Device where to do the training. None to choose cuda:0 if available,
        otherwise, cpu. (default: None)
    verbose : int
        The level of verbosity from 0 to 1. (default: 1)
    save_wts_to : str
        Name of a file to save the best model's weights. If None, then do not
        save. (default: None)
    save_model_to : str
        Name of a file to save the best whole model. If None, then do not save.
        (default: None)

    Returns
    -------
    best_model :
        The trained model with the lowest loss criterion during "val" phase
    """
    # get the device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the model to the device first
    model = model.to(device)

    if verbose >= 1:
        since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):
        if verbose >= 1:
            print("Epoch %d/%d" % (epoch+1, num_epochs))
            print("-"*10)

        # every epoch has a training and a validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                if scheduler is not None:
                    scheduler.step() # adjust the training learning rate
                model.train() # set the model to the training mode
            else:
                model.eval() # set the model to the evaluation mode

            # the total loss during this epoch
            running_loss = 0.0

            # iterate over the data
            dataset_size = 0
            for inputs, labels in dataloaders[phase]:
                # get the size of the dataset
                dataset_size += inputs.size(0)

                # load the inputs and the labels to the working device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # reset the model gradient to 0
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward gradient computation and optimize in training
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            # get the mean loss in this epoch
            epoch_loss = running_loss / float(dataset_size)

            if verbose >= 1:
                print("%s loss: %.4f" % (phase, epoch_loss))

            # copy the best model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

                # save the model
                if save_model_to is not None:
                    _save(model, save_model_to)
                if save_wts_to is not None:
                    _save(model.state_dict(), save_wts_to)
        print("")

    if verbose >= 1:
        time_elapsed = time.time()- since
        print("Training complete in %fs" % time_elapsed)
        print("Best val loss: %.4f" % best_loss)

    # load the best model
    model.load_state_dict(best_model_weights)
    return model

def _save(obj, fpath):
    torch.save(obj, fpath+".temp")
    try:
        os.rename(fpath+".temp", fpath)
    except OSError:
        if os.path.isfile(fpath):
            os.remove(fpath)
        if os.path.isdir(fpath):
            shutil.rmtree(fpath)
        os.rename(fpath+".temp", fpath)
