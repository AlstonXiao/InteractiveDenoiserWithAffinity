import torch
from torch import random

import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import denoiseNet, smapeLoss
from dataset import DenoiseDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
num_epochs = 2000
batch_size = 1
learning_rate = 1e-3

model = denoiseNet().to(device)
model.half()
for layer in model.modules():
  if isinstance(layer, nn.BatchNorm2d):
    layer.float()
    # print("layer corrected")
dataset = DenoiseDataset("D:/274 Traning Dataset")
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True)
n_total_steps = len(train_loader)


criterion = smapeLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.75)
AccLost = 0
epochLost = 0
for epoch in range(num_epochs):
    
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        
        labels = labels.to(device)

        # Forward pass
        with torch.set_grad_enabled(True):
            outputs,_ = model(images)
            loss = criterion.forward(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_lost = loss.to('cpu').item()
        AccLost += batch_lost
        epochLost += batch_lost
        if ((i+1) % 50 == 0):
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {AccLost:.4f}')
            AccLost = 0
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {epochLost:.8f}')
    step_lr_scheduler.step()
    AccLost = 0
    epochLost = 0
# store 
FILE = "D:/model.pth"
torch.save(model.state_dict(), FILE)
# load
# model = denoiseNet()
# model.load_state_dict(torch.load(PATH))
# model.to(device)

