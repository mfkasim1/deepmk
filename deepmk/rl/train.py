import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import deepmk.spv as spv
import deepmk.utils as mkutils

"""
This file contains method to train and test reinforcement learning model.
"""

__all__ = ["train", "show"]

def train(env, trainer, model, actor,
          reward_preproc=lambda x:x, scheduler=None, num_episodes=1000,
          val_every=20, val_episodes=10, verbose=1, plot=0,
          save_wts_to=None, save_model_to=None):

    """
    Performs a training of the model.

    Args:
        env :
            The RL environment, equivalent to OpenAI gym.
        trainer (deepmk.rl.trainers.Trainer) :
            The RL algorithm object that do the training
            every step and/or episode.
        model :
            A torch trainable class method that accepts "inputs" and returns
            prediction of "outputs".
        actor :
            An object that receives a state and returns an action
            recommendation.
        reward_preproc (function) :
            Reward preprocessor. (default: lambda x:x)
        scheduler (torch.optim.lr_scheduler object):
            Scheduler of how the learning rate is evolving through the epochs.
            If it is None, it does not update the learning rate. (default: None)
        num_episodes (int) :
            The number of episodes in training. (default: 1000)
        val_every (int) :
            Validate every given value of this argument. (default: 20)
        val_episodes (int) :
            The number of episodes averaged in the validation. (default: 10)
        verbose (int) :
            The level of verbosity from 0 to 1. (default: 1)
        plot (int) :
            Whether to plot the loss of training and validation data.
            (default: 0)
        save_wts_to (str) :
            Name of a file to save the best model's weights. If None, then do
            not save. (default: None)
        save_model_to (str) :
            Name of a file to save the best whole model. If None, then do not
            save. (default: None)

    Returns:
        best_model :
            The trained model with the best criterion during "val" phase.
    """
    # set interactive plot
    if plot:
        plt.ion()

    if verbose >= 1:
        since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    val_scores = []

    for episode in range(num_episodes):
        # get the phase
        phase = "val" if episode % val_every == 0 else "train"

        # set the appropriate model's mode
        if phase == "train":
            model.train()
        else:
            model.eval()

        nepisodes = 1 if phase == "train" else val_episodes
        score = 0
        for i in range(nepisodes):
            # apply the scheduler in training phase per episode
            if phase == "train" and scheduler is not None:
                scheduler.step()

            # starts for an episode
            state = env.reset()
            episode_done = False

            # iterate until the episode finishes
            while not episode_done:
                # get the next action
                action = actor.getaction(state)

                # act! and observe the state and reward
                next_state, reward, episode_done, _ = env.step(action)
                reward = reward_preproc(reward)
                score += reward

                # get the dataloader to train the model
                if phase == "train":
                    dataloader = trainer.trainstep(state, \
                        action, reward, next_state, \
                        episode_done)

                # update the state to the next state
                state = next_state

        # get the average score
        avg_score = score * 1. / nepisodes

        if phase == "val":
            val_scores.append(avg_score)

            for name, param in model.named_parameters():
                if param.requires_grad: print name, param.data

            # print the progress
            if verbose >= 1:
                print("Episode %d/%d: %f" % \
                      (episode+1, num_episodes, avg_score))

            # check if this is the best performance
            if avg_score > best_score:
                best_score = avg_score
                best_model_weights = copy.deepcopy(model.state_dict())

                # save the model
                if save_model_to is not None:
                    mkutils.save(model, save_model_to)
                if save_wts_to is not None:
                    mkutils.save(model.state_dict(), save_wts_to)

        # plot the validation scores
        if plot:
            xs_plot = range(0, len(val_scores)*val_every, val_every)
            plt.clf()
            plt.plot(xs_plot, val_scores, 'o-')
            plt.xlabel("Episode")
            plt.ylabel("Scores")
            plt.pause(0.001)

    if verbose >= 1:
        time_elapsed = time.time() - since
        print("Training complete in %fs" % time_elapsed)
        print("Best val score: %.4f" % best_score)

    # load the best model
    model.load_state_dict(best_model_weights)
    return model

def show(env, model, actor, load_wts_from=None,
         load_model_from=None):
    # load the model
    if load_model_from is not None:
        model = torch.load(load_model_from)
    elif load_wts_from is not None:
        model.load_state_dict(torch.load(load_wts_from))

    # play the episode
    while True:
        obs = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action = actor.getaction(obs)
            obs, reward, done, _ = env.step(action)
            score += reward
        print("Score: %f" % score)
