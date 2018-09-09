import deepmk
import torch
import gym
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from deepmk.rl.trainers import QLearn
from deepmk.rl.actors import QNet
from deepmk.rl import train, show

# set up the training components
env = gym.make("CartPole-v0")
model = nn.Sequential(
    nn.Linear(4, 100),
    nn.Sigmoid(),
    nn.Linear(100, env.action_space.n)
)
actor = QNet(model, epsilon=0.1)
optimizer = optim.SGD(model.parameters(),
    lr=1e-2, momentum=0.01)
trainer = QLearn(actor, optimizer, gamma=0.99)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# train the model
model = train(env, trainer, model, actor,
          reward_preproc=lambda x:x, scheduler=None, num_episodes=10000,
          val_every=20, val_episodes=10, verbose=1, plot=1,
          save_wts_to="cartpole1.pkl")

# show(env, model, actor, load_wts_from="cartpole.pkl")
