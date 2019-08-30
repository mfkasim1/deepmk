import torch
import torch.nn as nn
from torchvision import models

class Residual1d(torch.nn.Module):
    def __init__(self, ch, nunits,
                 batch_norm=False, layer_norm_shape=None,
                 skip=True,
                 last_activation=True):
        super(Residual1d, self).__init__()
        self.channels = ch
        self.nunits = nunits
        self.kernel_size = 3
        self.padding = 1
        self.batch_norm = batch_norm
        self.skip = skip
        self.last_activation = last_activation
        self.layer_norm_shape = layer_norm_shape

        # module units
        self.conv = torch.nn.ModuleList([
            torch.nn.Conv1d(
                in_channels = ch,
                out_channels = ch,
                kernel_size = self.kernel_size,
                padding = self.padding) \
            for i in range(nunits)])
        self.activation = torch.nn.ModuleList([
            torch.nn.ReLU() for i in range(nunits)])
        if self.batch_norm:
            self.bnmodules = torch.nn.ModuleList([
                torch.nn.BatchNorm1d(ch) for i in range(nunits)])
        if self.layer_norm_shape is not None:
            self.lnmodules = torch.nn.ModuleList([
                torch.nn.LayerNorm(layer_norm_shape) for i in range(nunits)])

    def forward(self, x):
        x0 = x.clone()
        for i in range(self.nunits):
            x = self.conv[i](x)
            if self.batch_norm:
                x = self.bnmodules[i](x)
            if self.layer_norm_shape is not None:
                x = self.lnmodules[i](x)
            if (i < self.nunits-1) or \
               ((i == self.nunits-1) and self.last_activation):
                x = self.activation[i](x)
        if self.skip:
            x = x + x0
        return x

class SelfAttn1d(torch.nn.Module):
    """
    This self attention module is different from self attention in SAGAN.
    This module split the channel into 3 parts:
    * f: the self attention part
    * g: another self attention part
    * h: the convolutional information part
    h would have `outch` numbers of channel and f and g would have the same
    number of channels (the remaining channels divided by 2 equally).
    f and g would act as the attention map to be multiplied with h.

    Arguments
    ---------
    * `outch` : int
        The number of output channel
    """
    def __init__(self, outch):
        super(SelfAttn1d, self).__init__()
        self.outch = outch
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, x):
        # check the channel
        ch = x.shape[-2]
        if (ch - self.outch) % 2 != 0:
            raise RuntimeError("The input channel - output channel must be "\
                               "an even number")

        # compute the attention
        hconv = x[:,:self.outch,:] # (nbatch, noc, nsig)
        fattn = x[:,self.outch::2,:].permute(0,2,1) # (nbatch, nsig, nc/2)
        gattn = x[:,self.outch+1::2,:] # (nbatch, nc/2, nsig)
        attn01 = torch.bmm(fattn, gattn) # (nbatch, nsig, nsig)
        attn = self.softmax(attn01)
        attnconv = torch.bmm(hconv, attn) # (nbatch, 1, nsig)
        res = attnconv * self.alpha + hconv * self.beta
        return res

class SelfAttn1d2(torch.nn.Module):
    def __init__(self, ch, och=None, with_conv=True):
        super(SelfAttn1d2, self).__init__()
        if och is None:
            och = 1 if ch // 8 < 1 else ch // 8

        self.with_conv = with_conv
        if with_conv:
            self.fconv = torch.nn.Conv1d(in_channels=ch, out_channels=och, kernel_size=1)
            self.gconv = torch.nn.Conv1d(in_channels=ch, out_channels=och, kernel_size=1)
            self.hconv = torch.nn.Conv1d(in_channels=ch, out_channels= ch, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, x):
        if self.with_conv:
            fconv = self.fconv(x) # (nbatch, och, nsig).permute(0,2,1) # (nbatch, nsig, och)
            gconv = self.gconv(x) # (nbatch, och, nsig)
            hconv = self.hconv(x) # (nbatch,  ch, nsig)
        else:
            fconv = x
            gconv = x
            hconv = x
        fgconf = torch.einsum("ji,ik->jk", fconv, gconv)
        fgconv = torch.bmm(fconv, gconv) # (nbatch, nsig, nsig)
        fgconv2 = self.softmax(fgconv)
        attn = torch.bmm(hconv, fgconv2) # (nbatch, ch, nsig)

        return self.gamma * attn + x
