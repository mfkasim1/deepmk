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
import deepmk.criteria as mkcriteria
import deepmk.utils as mkutils

name = "01-coco"

# transform the image to tensor
coco_both_transform = transforms.Compose([
    mktransforms.RandomCropTensor(320, pad_if_needed=True)
])
coco_img_transform = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# target transform to get the background as well
def target_transform(target):
    tgt_max, tgt_amax = target.max(dim=0)
    return torch.where(tgt_max > 0.0, tgt_amax+1, torch.zeros_like(tgt_amax))

fdir = os.path.dirname(os.path.abspath(__file__))
coco_dir = os.path.join(fdir, "..", "..", "dataset", "coco")

# get the coco dataset
coco = {
    x: mkdatasets.CocoDetection(os.path.join(coco_dir, "%s2017"%x),
        os.path.join(coco_dir, "annotations", ("instances_%s2017.json"%x)),
        both_transform=coco_both_transform,
        img_transform=coco_img_transform,
        target_transform=target_transform) # argmax over the channel
    for x in ["train", "val"]
}
# dataloader
dataloader = {
    x: DataLoader(coco[x], batch_size=16, shuffle=(x=="train"), num_workers=16)
    for x in ["train", "val"]
}

# construct the model
model = nn.Sequential(
    mkmodels.fcn8(coco["train"].ncategories+1), # plus the background
    nn.LogSoftmax(dim=1) # softmax over the channel
)
criteria = {
    # criterion: binary cross entropy with logits loss
    # (i.e. classification done per-pixel)
    "train": nn.NLLLoss(),
    "val": mkcriteria.IoU(last_layer="softmax", exclude_channels=0)
}

# the learning process
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

deepmk.spv.train(model, dataloader, criteria, optimizer,
    scheduler=scheduler, plot=0, save_wts_to=name+".pkl", verbose=2)

# deepmk.spv.validate(model, dataloader["val"], criteria["val"],
#     # load_wts_from=name+".pkl",
#     verbose=2)
