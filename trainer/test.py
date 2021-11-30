import torch
from torch import random
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import denoiseNet, smapeLoss
from dataset import DenoiseDataset
import cv2

def visualize_comparison(modelPath, dataPath, sampleIndex, device='cuda'):
    device = torch.device(device)
    model = denoiseNet()
    model.load_state_dict(torch.load(modelPath))
    model.half()
    model.to(device)

    sample = np.load(os.path.join(dataPath, str(sampleIndex) + "_traningSample.npy"), allow_pickle=True)
    ref = sample[0][:,8:136, 8:136]
    
    rad = torch.from_numpy(sample[1]).view(1,3,144,144)
    rad = rad[:,:,3:141, 3:141]

    unfold = torch.nn.Unfold(kernel_size = (11, 11))
    rad = (unfold(rad)).view(1,363,128,128)
    
    floats = sample[2]
    ints = sample[3].astype(np.float16)
    features = torch.from_numpy(np.concatenate([floats, ints],1)).view(1, 8, 17, 144, 144)
    modelInput = {"radiance" : rad, "features" : features}
    
    outputs,kernel = model(modelInput)
    # kernel = np.zeros([121,128,128])
    # kernel[60, :, :] = 1
    # kernel.reshape([1,121,128,128])
    # kernel = torch.from_numpy(kernel).to(device).view(1,121,128,128)
    # outputs = torch.zeros(1, 3, 128, 128).to(device)
    # rad = rad.to(device)
    # channelOutput = []
    # layerOutput = []
    # # print(kernel.shape)
    # for layer in range(kernel.shape[0]):
    #     for outputchannel in range(3):
    #         channelOutput.append(torch.sum((rad[layer][(outputchannel)*121:(outputchannel+1)*121] * kernel[layer]),0))
    #     layerOutput.append(torch.stack(channelOutput))
    # outputs = torch.stack(layerOutput)

    outputs = outputs.to('cpu')
    kernel  = kernel.to('cpu')
    outputs = outputs.detach().numpy().astype(np.float32)
    kernel  = kernel.detach().numpy().astype(np.float32)
    cv2.imshow("reference", cv2.resize(ref.astype(np.float32).transpose(1,2,0), (512, 512), fx=0, fy=0, interpolation = cv2.INTER_NEAREST))
    radiance = sample[1][:,8:136, 8:136]
    
    cv2.imshow("before denoise", cv2.resize(radiance.astype(np.float32).transpose(1,2,0), (512, 512), fx=0, fy=0, interpolation = cv2.INTER_NEAREST))
    normal = sample[2][1, 0:3,8:136, 8:136]
    cv2.imshow("before denoise normal", cv2.resize(normal.astype(np.float32).transpose(1,2,0), (512, 512), fx=0, fy=0, interpolation = cv2.INTER_NEAREST))
    albedo = sample[2][1, 3:6,8:136, 8:136]
    cv2.imshow("before denoise albedo", cv2.resize(albedo.astype(np.float32).transpose(1,2,0), (512, 512), fx=0, fy=0, interpolation = cv2.INTER_NEAREST))
    cv2.imshow("after denoise",cv2.resize(outputs[0].transpose(1,2,0), (512, 512), fx=0, fy=0, interpolation = cv2.INTER_NEAREST))
    kernel = kernel[0]*10
    # kernelAtMiddle = kernel.transpose(1,2,0)[60, 60].reshape(11,11)
    # cv2.imshow("kernel",cv2.resize(kernelAtMiddle, (512, 512), fx=0, fy=0, interpolation = cv2.INTER_NEAREST))

    cv2.waitKey(0)

FILE = "D:/model.pth"
if __name__ == "__main__":
    visualize_comparison(FILE, "D:/274 Traning Dataset", 1085)