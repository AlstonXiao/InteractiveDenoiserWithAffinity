import os
import struct
import torch
import torchvision
import random
import cv2
import numpy as np
from torch.utils.data import Dataset

class DenoiseDataset(Dataset):
    def __init__(self, path):
        # load the size of the files
        self.path = path
        self.testsamples = os.listdir(path)

    def __len__(self):
        return 1
        #return len(self.testsamples)
    
    def __getitem__(self, index):
        sample = np.load(os.path.join(self.path, str(1085+index) + "_traningSample.npy"), allow_pickle=True)
        ref = torch.from_numpy(sample[0][:,8:136, 8:136])
        rad = torch.from_numpy(sample[1])

        floats = sample[2]

        ints = sample[3].astype(np.float16)
        # print(floats.shape)
        # print(ints.shape)
        # print(type(floats))
        # print(type(ints))
        features = torch.from_numpy(np.concatenate([floats, ints],1))
        sample = {"radiance" : rad, "features" : features}
        return sample, ref

       