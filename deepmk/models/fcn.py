import torch
import torch.nn as nn
from torchvision import models

__all__ = ["fcn8", "fcn16", "fcn32"]

class FCNConv(torch.nn.Module):
    def __init__(self, nconv, inchannel, outchannel, kernel=3, padding=1):
        super(FCNConv, self).__init__()
        self.in_channel = inchannel
        self.out_channel = outchannel
        self.kernel_size = kernel
        self.padding = padding
        layers = []
        for i in range(nconv):
            ch0 = inchannel if i == 0 else outchannel
            ch1 = outchannel
            layers.append(nn.Conv2d(ch0, ch1, kernel, padding=padding))
            layers.append(nn.ReLU())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq.forward(x)

    def __call__(self, x):
        return self.forward(x)

def FCNPool(kernel=2, stride=2):
    return nn.MaxPool2d(kernel, stride=stride, ceil_mode=True)

def FCNUpsample(inchannel, outchannel, upsample):
    return nn.ConvTranspose2d(inchannel, outchannel, kernel_size=2*upsample,
            stride=upsample, padding=upsample//2)

def repsample(x, n): # only works for the power of 2s
    x2 = torch.cat((x, x), dim=-2)
    y = torch.cat((x2, x2), dim=-1)
    if n > 2:
        return repsample(y, n//2)
    else:
        return y

class fcn32(torch.nn.Module):
    def __init__(self, n_class):
        super(fcn32, self).__init__()
        # convolutional network
        self.conv1 = FCNConv(2, 3, 64)
        self.pool1 = FCNPool()
        self.conv2 = FCNConv(2, 64, 128)
        self.pool2 = FCNPool()
        self.conv3 = FCNConv(3, 128, 256)
        self.pool3 = FCNPool()
        self.conv4 = FCNConv(3, 256, 512)
        self.pool4 = FCNPool()
        self.conv5 = FCNConv(3, 512, 1024)
        self.pool5 = FCNPool()
        self.conv6 = FCNConv(1, 1024, 2048)
        self.drop6 = nn.Dropout2d()
        self.conv7 = FCNConv(1, 2048, 4096)
        self.drop7 = nn.Dropout2d()

        # upsample layers
        self.upsample32 = FCNUpsample(self.conv7.out_channel, n_class, 32)
        self.upsample16 = FCNUpsample(
            self.conv7.out_channel+self.conv4.out_channel, n_class, 16)
        self.upsample8  = FCNUpsample(
            self.conv7.out_channel+self.conv4.out_channel+self.conv3.out_channel,
            n_class, 8)

        # softmax class
        self.softmax = nn.Softmax2d()

    def forward(self, inp):
        conv1 = self.conv1(inp)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        pool5 = self.pool5(conv5)

        conv6 = self.conv6(pool5)
        drop6 = self.drop6(conv6)
        conv7 = self.conv7(drop6)
        drop7 = self.drop7(conv7)

        res = self.upsample32(drop7)
        return res

class fcn16(fcn32):
    def forward(self, inp):
        conv1 = self.conv1(inp)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        pool5 = self.pool5(conv5)

        conv6 = self.conv6(pool5)
        drop6 = self.drop6(conv6)
        conv7 = self.conv7(drop6)
        drop7 = self.drop7(conv7)

        # get the input to upsample
        drop7_2x = repsample(drop7, 2)
        h = torch.cat((pool4, drop7_2x), dim=1)
        res = self.upsample16(h)
        return res

class fcn8(fcn32):
    def forward(self, inp):
        conv1 = self.conv1(inp)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        pool5 = self.pool5(conv5)

        conv6 = self.conv6(pool5)
        drop6 = self.drop6(conv6)
        conv7 = self.conv7(drop6)
        drop7 = self.drop7(conv7)

        # get the input to upsample
        drop7_4x = repsample(drop7, 4)
        pool4_2x = repsample(pool4, 2)
        h = torch.cat((pool3, pool4_2x, drop7_4x), dim=1)
        res = self.upsample8(h)
        return res
