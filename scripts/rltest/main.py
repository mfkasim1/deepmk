import deepmk
import torch
import gym
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from deepmk.rl.algs import MonteCarloRL
from deepmk.rl.actors import QNet
from deepmk.rl import train, show

# set up the training components
env = gym.make("CartPole-v0")
rlalg = MonteCarloRL(
    dataloader_kwargs={"shuffle": False, "batch_size": 1, "num_workers": 1},
    state_transform=torch.FloatTensor,
)
model = nn.Sequential(
    nn.Linear(4, 100),
    nn.Sigmoid(),
    nn.Linear(100, env.action_space.n)
)
actor = QNet(model, gamma=0.99)
optimizer = optim.SGD(model.parameters(),
    lr=1e-2, momentum=0.01)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# train the model
model = train(env, rlalg, model, actor, optimizer,
          reward_preproc=lambda x:x, scheduler=None, num_episodes=10000,
          val_every=20, val_episodes=10, verbose=1, plot=1,
          save_wts_to="cartpole.pkl", save_model_to=None)

# show(env, model, actor, load_wts_from="cartpole.pkl")
