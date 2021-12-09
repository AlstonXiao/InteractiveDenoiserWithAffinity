import os
import torch
import numpy as np
from torch.utils.data import Dataset

# Each training sample 
# sample[0] is reference, npfloat32
# sample[1] is image need to be denoise, npfloat32
# sample[2] is the float input data, npfloat16
# sample[3] is the boolean input data, npbool
# sample[4] is the pre-optix reference, npfloat32
# denoise will control the reference image be the pre-optix or post optix reference

class DenoiseDataset(Dataset):
    def __init__(self, path, denoise = True):
        # load the size of the files
        self.path = path
        self.testsamples = os.listdir(path)
        self.denoise = denoise

    def __len__(self):
        # return 1
        return len(self.testsamples)
    
    def __getitem__(self, index):
        sample = np.load(os.path.join(self.path, str(index) + "_traningSample.npy"), allow_pickle=True)
        ref = torch.from_numpy(sample[0][:,8:136, 8:136])
        if not self.denoise:
            ref = torch.from_numpy(sample[4][:,8:136, 8:136])

        rad = torch.from_numpy(sample[1]).view(1,3,144,144)
        rad = rad[:,:,3:141, 3:141]
        unfold = torch.nn.Unfold(kernel_size = (11, 11))
        rad = (unfold(rad)).view(1,363,128,128)
        rad = rad[0]

        floats = sample[2].astype(np.float32)
        ints = sample[3].astype(np.float32)

        features = torch.from_numpy(np.concatenate([floats, ints],1))
        sample = {"radiance" : rad, "features" : features}
        return sample, ref

       