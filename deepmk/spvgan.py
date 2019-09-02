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
This file contains method to train and test supervised learning + GAN.
"""

__all__ = ["train"]

def get_weights(*models):
    return copy.deepcopy([m.state_dict() for m in models])

def train(m_model, g_model, d_model,
          z_size, dataloaders,
          m_opt, g_opt, d_opt,
          mg_opt=None,
          m_sched=None, g_sched=None, d_sched=None,
          gan_criteria="hinge", spv_criteria="mse",
          num_epochs=25, device=None, verbose=1, plot=0,
          save_wts_to=None):
    """
    Performs a supervised + GAN training procedure. The generative and
    discriminative models are trained with GAN procedure while the mapper
    is trained with supervised procedure.
    In making the prediction, `m_model` is concatenated with `g_model` to
    generate signal from a given set of parameters.

    In one training batch:
    * `d_model` is trained by maximizing d-score for real and minimizing for
                fake signal.
    * `g_model` is trained by maximizing d-score for its generated signal.
    * `m_model` is trained by minimizing loss value from the dataset.

    Args:
        m_model :
            A torch trainable mapper model to map from parameters space to the
            latent space.
        g_model :
            A torch trainable generative model from the latent space to the
            signal space.
        d_model :
            A torch trainable discriminative model that receives the signal
            as the input and gives low score for fake and high score for real.
        z_size (int) :
            The size of the latent variables for the generative model.
        dataloaders (dict or torch.utils.data.DataLoader):
            Dictionary with two keys: ["train", "val"] with every value is an
            iterable with two outputs: (1) the "inputs" to the model and (2) the
            ground truth of the "outputs". If it is a DataLoader, then it's only
            for the training, nothing for validation.
        m_opt, g_opt, d_opt (torch.optim optimizer):
            Optimizer class in training the m_model, g_model, d_model, resp.
        mg_opt (torch.optim optimizer):
            Optimizer class to train the generative model in minimizing the
            loss spv function.
        m_sched, g_sched, d_sched (torch.optim.lr_scheduler object):
            Optimizer scheduler in training the m_model, g_model, d_model, resp.
            Default: None.
        gan_criteria (str, optional):
            Criteria in training GAN. For now, the option is only "hinge".
            Default: "hinge".
        spv_criteria (str,optional):
            Criteria in the supervised training. For now, the option is only
            "mse". Default: "mse".
        num_epochs (int):
            The number of epochs in training. (default: 25)
        device :
            Device where to do the training. None to choose cuda:0 if available,
            otherwise, cpu. (default: None)
        verbose (int):
            The level of verbosity from 0 to 1. (default: 1)
        save_wts_to (str):
            Name of a file to save the best model's weights. If None, then do not
            save. (default: None)

    Returns:
        best_model :
            The trained model with the lowest loss criterion during "val" phase
    """
    lambda_gp = 10.0

    # get the device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:")
    print(device)

    # check if the dataloader is for validation as well
    if type(dataloaders) != dict:
        dataloaders = {"train": dataloaders, "val": []}

    # load the model to the device first
    m_model = m_model.to(device)
    g_model = g_model.to(device)
    d_model = d_model.to(device)

    if verbose >= 1:
        since = time.time()
    best_model_weights = get_weights(m_model, g_model, d_model)
    best_loss = np.inf

    train_losses = []
    val_losses = []

    # book keeping scores
    d_loss_real_mean = {"train":0.0, "val":0.0}
    d_loss_fake_mean = {"train":0.0, "val":0.0}
    g_loss_mean = {"train":0.0, "val":0.0}
    m_loss_mean = {"train":0.0, "val":0.0}

    total_batches = len(dataloaders["train"]) + len(dataloaders["val"])
    for epoch in range(num_epochs):
        if verbose >= 1:
            print("Epoch %d/%d" % (epoch+1, num_epochs))
            print("-"*10)

        if verbose >= 2:
            progress_disp = mkutils.ProgressDisplay()

        # to time the progress
        epoch_start_time = time.time()

        # progress counter
        num_batches = 0 # num batches in training and validation
        if verbose >= 2:
            progress_disp = mkutils.ProgressDisplay()

        # every epoch has a training and a validation phase
        for phase in ["train", "val"]:

            # skip phase if the dataloaders for the current phase is empty
            if dataloaders[phase] == []: continue

            # set the model's mode
            if phase == "train":
                if m_sched is not None:
                    m_sched.step() # adjust the training learning rate
                if g_sched is not None:
                    g_sched.step()
                if d_sched is not None:
                    d_sched.step()
                m_model.train() # set the model to the training mode
                g_model.train()
                d_model.train()
            else:
                m_model.eval() # set the model to the evaluation mode
                g_model.eval()
                d_model.eval()

            # book keeping score
            d_loss_real_total = 0.0
            d_loss_fake_total = 0.0
            g_loss_total = 0.0
            m_loss_total = 0.0
            ndata_total = 0
            for params, signal in dataloaders[phase]:
                # write the progress bar
                num_batches += 1
                if verbose >= 2:
                    progress_disp.show(num_batches, total_batches)

                batch_size = params.shape[0]
                ndata = batch_size
                ndata_total += ndata

                # load to device
                params = params.to(device)
                signal = signal.to(device)

                ################ train the discriminator ################
                # calculate the d-scores for real and fake signals
                d_score_real = d_model(signal)
                z = torch.randn((signal.shape[0], z_size)).to(device)
                fake_signal = g_model(z)
                d_score_fake = d_model(fake_signal.detach())

                # maximizing score for the real signal
                # minimizing score for the fake signal
                if gan_criteria == "hinge":
                    d_loss_real = torch.clamp(1.0 - d_score_real, 0.0).mean()
                    d_loss_fake = torch.clamp(1.0 + d_score_fake, 0.0).mean()
                elif gan_criteria == "wgan-gp":
                    d_loss_real = -d_score_real.mean()
                    d_loss_fake =  d_score_fake.mean()
                elif gan_criteria == "bce":
                    real_label = torch.full((batch_size,), 1, device=device)
                    fake_label = torch.full((batch_size,), 0, device=device)
                    d_loss_real = torch.nn.BCELoss()(d_score_real, real_label)
                    d_loss_fake = torch.nn.BCELoss()(d_score_fake, fake_label)

                # backprop the discriminator
                d_loss = d_loss_fake + d_loss_real
                if phase == "train":
                    d_model.zero_grad()
                    d_opt.zero_grad()
                    d_loss.backward()
                    d_opt.step()

                if gan_criteria == "wgan-gp":
                    alpha = torch.rand(signal.shape[0],1,1).to(device).expand_as(signal)
                    interpolated = torch.zeros_like(signal)
                    interpolated.data = alpha * signal.data + (1-alpha) * fake_signal.data
                    interpolated.requires_grad = True

                    d_interp = d_model(interpolated)
                    grad = torch.autograd.grad(
                        outputs=d_interp,
                        inputs=interpolated,
                        grad_outputs=torch.ones(d_interp.size()).cuda(),
                        retain_graph=True,
                        create_graph=True,
                        only_inputs=True)[0]

                    grad = grad.view(grad.size(0), -1)
                    grad_l2norm = torch.sqrt(torch.sum(grad*grad, dim=1))
                    d_loss_gp = torch.mean((grad_l2norm-1)**2)

                    # backward + optimize
                    d_loss = lambda_gp * d_loss_gp
                    d_opt.zero_grad()
                    d_loss.backward()
                    d_opt.step()

                # book keeping
                d_loss_real_total += d_loss_real.data * ndata
                d_loss_fake_total += d_loss_fake.data * ndata

                ################ train the generator ################
                # generate fake signal
                d_score_fake = d_model(fake_signal)

                # maximize the d-score for the fake signal
                if gan_criteria in ["hinge", "wgan-gp"]:
                    g_loss = -d_score_fake.mean()
                elif gan_criteria == "bce":
                    g_loss = torch.nn.BCELoss()(d_score_fake, real_label)

                # backprop the generative model
                if phase == "train":
                    g_model.zero_grad()
                    g_opt.zero_grad()
                    g_loss.backward()
                    g_opt.step()

                # clear the memory
                del z

                # book keeping
                g_loss_total += g_loss.data * ndata

                ################ train the mapper ################
                # get the signal from the parameters
                predict_signal = g_model(m_model(params))

                # calculate the loss function
                if spv_criteria == "mse":
                    sig_err = (predict_signal - signal)
                    m_loss = (sig_err * sig_err).mean()

                # backprop the mapper model
                if phase == "train":
                    m_model.zero_grad()
                    m_opt.zero_grad()
                    if mg_opt is not None:
                        g_model.zero_grad()
                        mg_opt.zero_grad()
                        g_opt.zero_grad()

                    m_loss.backward()
                    m_opt.step()
                    if mg_opt is not None:
                        mg_opt.step()


                # book keeping
                m_loss_total += m_loss.data * ndata

            # finish one part of the epoch (either train or val)
            # get the mean values
            d_loss_real_mean[phase] = d_loss_real_total / ndata_total
            d_loss_fake_mean[phase] = d_loss_fake_total / ndata_total
            g_loss_mean[phase] = g_loss_total / ndata_total
            m_loss_mean[phase] = m_loss_total / ndata_total

            # copy the best model
            if phase == "val" and m_loss_mean[phase] < best_loss:
                best_loss = m_loss_mean[phase].data
                best_model_weights = get_weights(m_model, g_model, d_model)

                # save the model
                if save_wts_to is not None:
                    mkutils.save(best_model_weights, save_wts_to)

        # finish one epoch

        # print the message
        if verbose > 0:
            print("Done in %fs (best val loss: %.3e)" % (time.time() - since, best_loss))
            print("D-loss real: (train) %.3e, (val) %.3e" % \
                (d_loss_real_mean["train"], d_loss_real_mean["val"]))
            print("D-loss fake: (train) %.3e, (val) %.3e" % \
                (d_loss_fake_mean["train"], d_loss_fake_mean["val"]))
            print("G-loss fake: (train) %.3e, (val) %.3e" % \
                (g_loss_mean["train"], g_loss_mean["val"]))
            print("M-loss     : (train) %.3e, (val) %.3e" % \
                (m_loss_mean["train"], m_loss_mean["val"]))

    # finish all epochs
    if verbose >= 1:
        time_elapsed = time.time() - since
        print("Training complete in %fs" % time_elapsed)
        print("Best val loss: %.4f" % best_loss)

    # load the best models
    m_model.load_state_dict(best_model_weights[0])
    g_model.load_state_dict(best_model_weights[1])
    d_model.load_state_dict(best_model_weights[2])
    return m_model, g_model, d_model, best_loss
