import os
import struct
import torch
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
        return 4
        #return len(self.testsamples)
    
    def __getitem__(self, index):
        sample = np.load(os.path.join(self.path, str(1085+index) + "_traningSample.npy"), allow_pickle=True)
        ref = torch.from_numpy(sample[0][:,8:136, 8:136])
        rad = torch.from_numpy(sample[1]).view(1,3,144,144)
        rad = rad[:,:,3:141, 3:141]
        unfold = torch.nn.Unfold(kernel_size = (11, 11))
        rad = (unfold(rad)).view(1,363,128,128)
        rad = rad[0]
        floats = sample[2]
        ints = sample[3].astype(np.float16)
        # print(floats.shape)
        # print(ints.shape)
        # print(type(floats))
        # print(type(ints))
        features = torch.from_numpy(np.concatenate([floats, ints],1))
        sample = {"radiance" : rad, "features" : features}
        return sample, ref

       