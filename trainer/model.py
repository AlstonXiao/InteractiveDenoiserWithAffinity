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
        self.embedding_width =32

        # Unet
        self.unet = UNet(self.embedding_width, 9, [64,64,80,96])

    def forward(self, samples):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = 'cpu'
        radiance = samples["radiance"]
        features = samples["features"]
        radiance = radiance.to(device)
        features = features.to(device)

        bs, spp, nf, h, w = features.shape

        features = features.view([bs*spp, nf, h, w])
        features = self.FC1perSample(features)
        features = F.leaky_relu(features)
        features = self.FC2perSample(features)
        features = F.leaky_relu(features)
        features = self.FC3perSample(features)
        features = F.leaky_relu(features)

        features = features.view(bs, spp, self.embedding_width, h, w)
        reduced = features.mean(1)
        kernel = self.unet(reduced)
        
        # apply
        # kernel = kernel[:, :,8:136,8:136 ]
        # kernel = F.softmax(kernel, dim=1)
        # channel = 3
        
        # for layer in range(kernel.shape[0]):
        
        # output = torch.zeros(bs, channel, 128, 128).to(device)
        # output.
        # channelOutput = []
        # layerOutput = []
        # # print(kernel.shape)
        # for layer in range(kernel.shape[0]):
        #     for outputchannel in range(channel):
        #         channelOutput.append(torch.sum((radiance[layer][(outputchannel)*121:(outputchannel+1)*121] * kernel[layer]),0))
        #     layerOutput.append(torch.stack(channelOutput))
            
        # return torch.stack(layerOutput), kernel
        # return kernel[:, :,8:136,8:136 ], kernel
        
        # kernel = f0, a0, c0, f1, a1, ...
        
        for KK in range(3):
            output = []
            for layer in range(kernel.shape[0]):
                imageColor1 = []
                imageColor2 = []
                imageColor3 = []
                bs, nf, h, w = radiance.shape
                for i in range(KK+1, h-KK-1):
                    rowColor1 = []
                    rowColor2 = []
                    rowColor3 = []
                    for j in range(KK+1, w-KK-1):
                        weightSum = []
                        pixelColor1 = []
                        pixelColor2 = []
                        pixelColor3 = []
                        for k in range(-1, 1):
                            for a in range(-1, 1):
                                weight = torch.exp(-kernel[layer][KK*3 + 1][i][j] * torch.pow((kernel[layer][KK*3 +0][i+k*KK][j+a*KK] - kernel[layer][KK*3 +0][i][j]), torch.tensor(2).to(device)))
                                if k == 0 and a == 0:
                                    weight = kernel[layer][KK*3 +2][i][j]
                                
                                weightSum.append(weight)
                                pixelColor1.append(weight * radiance[layer][0][i+k][j+a])
                                pixelColor2.append(weight * radiance[layer][2][i+k][j+a])
                                pixelColor3.append(weight * radiance[layer][1][i+k][j+a])
                        totalWeight = torch.stack(weightSum).mean()
                        rowColor1.append(torch.stack(pixelColor1).sum() / totalWeight)
                        rowColor2.append(torch.stack(pixelColor2).sum() / totalWeight)
                        rowColor3.append(torch.stack(pixelColor3).sum() / totalWeight)
                    imageColor1.append(torch.stack(rowColor1))
                    imageColor2.append(torch.stack(rowColor2))
                    imageColor3.append(torch.stack(rowColor3))
                output.append(torch.stack([torch.stack(imageColor1), torch.stack(imageColor2), torch.stack(imageColor3)]))
            radiance = torch.stack(output)

        e = 2
        return radiance[:, :, 2:130, 2:130], kernel






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
