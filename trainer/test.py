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
    outputs = model(modelInput)
    outputs = outputs.to('cpu')
    outputs = outputs.detach().numpy().astype(np.float32)
    cv2.imshow("reference", cv2.resize(ref.astype(np.float32).transpose(1,2,0), (512, 512)))
    radiance = sample[1][:,8:136, 8:136]
    cv2.imshow("before denoise", cv2.resize(radiance.astype(np.float32).transpose(1,2,0), (512, 512)))
    normal = sample[2][1, 6:9,8:136, 8:136]
    cv2.imshow("before denoise normal", cv2.resize(normal.astype(np.float32).transpose(1,2,0), (512, 512)))
    albedo = sample[2][1, 9:12,8:136, 8:136]
    cv2.imshow("before denoise albedo", cv2.resize(albedo.astype(np.float32).transpose(1,2,0), (512, 512)))
    cv2.imshow("after denoise",cv2.resize(outputs[0].transpose(1,2,0), (512, 512)))
    cv2.waitKey(0)

FILE = "D:/model.pth"
if __name__ == "__main__":
    visualize_comparison(FILE, "D:/274 Traning Dataset", 1086)