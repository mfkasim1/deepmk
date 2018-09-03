"""
This is modified file from PyTorch tutorial to take the advantage of deepmk.
"""

import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import deepmk

# data directory
fdir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(fdir, "..", "..", "dataset", "hymenoptera_data")

# transformations of the data
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# dataset where the data should be loaded
image_datasets = {
    x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                            transform=data_transforms[x])
    for x in ["train", "val"]
}

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                   shuffle=True, num_workers=4)
    for x in ["train", "val"]
}

# load the pretrained resnet18
model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False

# replace the fully connected layer to a new layer with 2 outputs
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

# cross entropy loss for multi classes classifications
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)

# reduce the learning rate by 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# train the model
model_ft = deepmk.spv.train(model_ft, dataloaders, criterion, optimizer_ft,
    scheduler=exp_lr_scheduler, num_epochs=25, plot=1)
