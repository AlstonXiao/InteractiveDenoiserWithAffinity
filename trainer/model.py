import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class denoiseNet(nn.Module):
    def __init__(self, final_layer_size = 11, kernel_size = 11,  embedding_width = 32, unet_channel_structure = [64, 64, 80, 96],):
        super(denoiseNet, self).__init__()
        self.FC1perSample = nn.Conv2d(17, 32, 1, 1)
        self.FC2perSample = nn.Conv2d(32, 32, 1, 1) 
        self.FC3perSample = nn.Conv2d(32, 32, 1, 1) 
        self.embedding_width = embedding_width
        self.kernel_total_size = kernel_size*kernel_size
        # Unet
        self.unet = UNet(self.embedding_width, self.kernel_total_size, unet_channel_structure, final_layer_size)

    def forward(self, samples):
        radiance = samples["radiance"]
        features = samples["features"]
        radiance = Variable(radiance)
        features = Variable(features)

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
        kernel = kernel[:, :,8:136,8:136 ]
        kernel = F.softmax(kernel, dim=1)
        channel = 3

        # we are operating on floating point images
        kernel = kernel.float()
        layerOutput = []
        # print(kernel.shape)
        for layer in range(kernel.shape[0]):
            channelOutput = []
            for outputchannel in range(channel):
                channelOutput.append(torch.sum((radiance[layer][(outputchannel)*self.kernel_total_size:(outputchannel+1)*self.kernel_total_size] * kernel[layer]),0))
            layerOutput.append(torch.stack(channelOutput))
            
        return torch.stack(layerOutput), kernel

class UNet(nn.Module):
    def __init__(self, input_chan, output_chan, dims, finalLayer):
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
        self.outputLayer = nn.Conv2d(dims[0], output_chan, kernel_size=finalLayer, padding=finalLayer//2)
            

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
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=padding, padding_mode="zeros"),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=padding, padding_mode="zeros"),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.doubelConvolution(x)
