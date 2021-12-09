import torch
import torch.nn as nn
import numpy as np

class smapeLoss(nn.Module):
    def __init__(self, eps=1e-4):
        super(smapeLoss, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        loss = (torch.abs(im-ref) / (
            self.eps + torch.abs(im.detach()) + torch.abs(ref.detach()))).mean()
        return loss/3

def smape_np(im, ref,  eps=1e-2):
    loss = np.mean(np.absolute(im - ref) / (eps +np.absolute(im) + np.absolute(ref)))
    return loss/3

def l1_np(im, ref):
    return np.mean(np.absolute(im - ref))