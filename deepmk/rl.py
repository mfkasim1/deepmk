import torch
import torch.nn as nn
import deepmk.spv

"""
This file contains method to train and test reinforcement learning model.
"""

__all__ = ["train"]

def train(env, rlalg, model, actor, optimizer, criterion=nn.MSELoss, scheduler=None,
          num_episodes=1000, val_every=20, val_episodes=10, verbose=1, plot=0,
          save_wts_to=None, save_model_to=None):

    """
    Performs a training of the model.

    Args:
        env :
            The RL environment, equivalent to OpenAI gym.
        rlalg (deepmk.rlalg.RLAlg) :
            The RL algorithm object.
        model :
            A torch trainable class method that accepts "inputs" and returns
            prediction of "outputs".
        actor :
            An object that receives a state and returns an action
            recommendation.
        optimizer (torch.optim optimizer) :
            Optimizer class in training the model.
        criterion (function or evaluable class) :
            Receives the prediction of the "outputs" as the first argument, and
            the ground truth of the "outputs" as the second argument. It returns
            the loss function to be minimized. (default: torch.nn.MSELoss)
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
    # get the device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            # starts for an episode
            state = env.reset()
            episode_done = False

            # iterate until the episode finishes
            while not episode_done:
                # get the next action
                action = actor(state)

                # act! and observe the state and reward
                next_state, reward, episode_done, _ = env.step(action)
                score += reward
                if episode_done: next_state = None

                # get the dataloader to train the model
                dataloader = rlalg.step(state, action, reward, next_state)
                spv.train(model, dataloader, criterion, optimizer, scheduler,
                    num_epochs=1, verbose=0, plot=0)

        # get the average score
        avg_score = score * 1. / nepisodes

        if phase == "val":
            val_scores.append(avg_score)

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

        print("")

    if verbose >= 1:
        time_elapsed = time.time() - since
        print("Training complete in %fs" % time_elapsed)
        print("Best val score: %.4f" % best_score)

    # load the best model
    model.load_state_dict(best_model_weights)
    return model
