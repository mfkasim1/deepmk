import os
import numpy as np
import torch
import deepmk
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import deepmk.datasets as mkdatasets
import deepmk.models as mkmodels
import deepmk.transforms as mktransforms

name = "01-coco"

# transform the image to tensor
coco_both_transform = transforms.Compose([
    mktransforms.RandomCropTensor(320, pad_if_needed=True)
])
fdir = os.path.dirname(os.path.abspath(__file__))
coco_dir = os.path.join(fdir, "..", "..", "dataset", "coco")
# get the coco dataset
coco = {
    x: mkdatasets.CocoDetection(os.path.join(coco_dir, "%s2017"%x),
        os.path.join(coco_dir, "annotations", ("instances_%s2017.json"%x)),
        both_transform=coco_both_transform)
    for x in ["train", "val"]
}
# dataloader
dataloader = {
    x: DataLoader(coco[x], batch_size=64, shuffle=True, num_workers=16)
    for x in ["train", "val"]
}

# load the model
fcn = mkmodels.fcn8(coco["train"].ncategories)

# criterion: binary cross entropy with logits loss
# (i.e. classification done per-pixel)
criterion = nn.BCEWithLogitsLoss()

# the learning process
optimizer = optim.SGD(fcn.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# train(model, dataloaders, criterion, optimizer, scheduler=None,
#           num_epochs=25, device=None, verbose=1, plot=0, save_wts_to=None,
#           save_model_to=None)
deepmk.spv.train(fcn, dataloader, criterion, optimizer,
    scheduler=scheduler, plot=0, save_wts_to=name+".pkl", verbose=2)
