import torch
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.subplots_adjust(bottom=0.65)
    plt.show()

def model_to_half(model):
    model.half()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

def plot_training_curve(input_json):
    f = open(input_json)
    data = json.load(f)

    for i in range(len(data["epoch_ssim"])):
        data["epoch_ssim"][i] = 1-data["epoch_ssim"][i]
        data["epoch_Vssim"][i] = 1-data["epoch_Vssim"][i]
    plt.plot(data["epoch_smape"], color="b")
    plt.plot(data["epoch_Vsmape"],  color="g")
    plt.plot(data["epoch_ssim"], color="r")
    plt.plot(data["epoch_Vssim"],  color="y")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.title("SMAPE loss and 1-SSIM")
    plt.show()
    
    f.close()

if __name__ == "__main__":
    plot_training_curve("C:/Users/Alsto/OneDrive - UC San Diego/CSE 274/training log/7Full1NewDBmodel.json")