import torch
from torch import random

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class smapeLoss(nn.Module):
    def __init__(self, eps=1e-4):
        super(smapeLoss, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        loss = (torch.abs(im-ref) / (
            self.eps + torch.abs(im.detach()) + torch.abs(ref.detach()))).mean()
        return loss/3

class denoiseNet(nn.Module):
    def __init__(self):
        super(denoiseNet, self).__init__()
        self.FC1perSample = nn.Conv2d(17, 32, 1, 1)
        self.FC2perSample = nn.Conv2d(32, 32, 1, 1) 
        self.FC3perSample = nn.Conv2d(32, 32, 1, 1) 
        self.embedding_width = 32

        # Unet
        self.unet = UNet(32, 11*11, [64,64,80,96])

    def forward(self, samples):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = 'cpu'
        radiance = samples["radiance"]
        features = samples["features"]
        radiance = radiance.to(device)
        features = features.to(device)

        bs, spp, nf, h, w = features.shape
        

        flat = features.view([bs*spp, nf, h, w])
        flat = self.FC1perSample(flat)
        flat = F.leaky_relu(flat)
        flat = self.FC2perSample(flat)
        flat = F.leaky_relu(flat)
        flat = self.FC3perSample(flat)
        flat = F.leaky_relu(flat)

        flat = flat.view(bs, spp, self.embedding_width, h, w)
        reduced = flat.mean(1)
        kernel = self.unet(reduced)
        
        # apply
        kernel = kernel[:, :,8:136,8:136 ]
        kernel = F.softmax(kernel, dim=1)
        channel = 3
        
        # for layer in range(kernel.shape[0]):
        
        output = torch.zeros(bs, channel, 128, 128).to(device)
        # print(kernel.shape)
        for layer in range(kernel.shape[0]):
            for outputchannel in range(channel):
                output[layer][outputchannel] = (radiance[layer][(outputchannel)*121:(outputchannel+1)*121] * kernel[layer]).sum(0)
        return output 
        # return kernel[:, :,8:136,8:136 ]



class UNet(nn.Module):
    def __init__(self, input_chan, output_chan, dims):
        super(UNet, self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pooling = nn.MaxPool2d(2,2)

        for dim in dims[:-1]:
            self.down.append(doubleCov(input_chan, dim, 1))
            input_chan = dim
        # btn
        self.bottleneck = doubleCov(dims[-2], dims[-1], 1)
        # ups
        last = dims[-1]
        for dim in (reversed(dims[:-1])):
            self.up.append(
                nn.ConvTranspose2d(last, dim, 2, 2)
            )
            self.up.append(doubleCov(dim*2, dim, 1))
            last = dim
        self.outputLayer = nn.Conv2d(dims[0], output_chan, kernel_size=1)
            

    def forward(self, x):
        skip = []
        for down in self.down:
            x = down(x)
            skip.append(x)
            x = self.pooling(x)

        x = self.bottleneck(x)
        skip = skip[::-1]
        for i in range(0, len(self.up), 2):
            x = self.up[i](x)
            skipped = skip[i//2]
            x = self.up[i+1](torch.cat((x, skipped), dim=1))
        x = self.outputLayer(x)
        return x


class doubleCov(nn.Module):
    def __init__(self, input_channel, output_channel, padding):
        super(doubleCov, self).__init__()
        self.doubelConvolution = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=padding, padding_mode="reflect"),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=padding, padding_mode="reflect"),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.doubelConvolution(x)

# inputs = {"features" : torch.rand(4,8,17,144,144), "radiance" : torch.rand(4, 3, 144, 144) }
# model = denoiseNet()
# print(model.forward(inputs)[0][0])
