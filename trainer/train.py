import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import numpy as np
from torchviz import make_dot
from datetime import datetime
from pytorch_msssim import ssim
from piqa import psnr
import json

from model import denoiseNet
from metrics import smapeLoss
from dataset import DenoiseDataset
from utils import plot_grad_flow


def train_model(model, criterion, optimizer, scheduler, training_detail, dataset, output_path, log_path, device):

    num_epochs = training_detail['num_epochs']
    batch_size = training_detail['batch_size']
    train_test_ratio = 0.8
    if 'train_test_ratio' in training_detail: train_test_ratio = training_detail['train_test_ratio']

    train_size = int(train_test_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    a = len(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    n_total_steps = len(train_loader)

    best_acc = 1000.0

    print('Start training')
    f = open(log_path+".txt", "w")
    f.write('Starting at: %s\n' %datetime.now())

    epoch_smape = []
    epoch_l1 = []
    epoch_l2 = []
    epoch_ssim = []

    epoch_Vsmape = []
    epoch_Vl1 = []
    epoch_Vl2 = []
    epoch_Vssim = []

    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    for epoch in range(num_epochs):
        Accumulated_loss_50_batch = 0
        epochTrainSMAPELost = 0
        epochValSMAPELost = 0

        epochTrainL1Lost = 0
        epochValL1Lost = 0

        epochTrainL2Lost = 0
        epochValL2Lost = 0

        epochTrainSSIMLost = 0
        epochValSSIMLost = 0
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = test_loader

            for i, (images, labels) in enumerate(dataloader):
                labels = labels.to(device)
                images["radiance"] = images["radiance"].to(device)
                images["features"] = images["features"].to(device)
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs,_ = model((images))
                    loss = criterion.forward(outputs, labels)
                    # outputs.to('cpu')
                    # dot = make_dot(outputs, params=dict(model.named_parameters()))
                    # dot.view()
                    # outputs.to(device)

                    # Backward and optimize
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        # model.to('cpu')
                        # plot_grad_flow(model.named_parameters())
                        # model.to(device)
                        optimizer.step()

                batch_lost = loss.to('cpu').item()
                Accumulated_loss_50_batch += batch_lost
                l1batch = l1loss(outputs, labels).to('cpu').item()
                l2batch = l2loss(outputs, labels).to('cpu').item()
                ssimBatch = ssim(outputs, labels, data_range=1, size_average=True).to('cpu').item()
                if phase == 'train':
                    epochTrainSMAPELost += batch_lost * labels.size(0)
                    epochTrainL1Lost += l1batch  * labels.size(0)
                    epochTrainL2Lost+= l2batch  * labels.size(0)
                    epochTrainSSIMLost+= ssimBatch  * labels.size(0)
                else:
                    epochValSMAPELost += batch_lost  * labels.size(0)
                    epochValL1Lost += l1batch * labels.size(0)
                    epochValL2Lost += l2batch  * labels.size(0)
                    epochValSSIMLost += ssimBatch * labels.size(0)

                if ((i+1) % 50 == 0) and phase == 'train':
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {Accumulated_loss_50_batch:.4f}')
                    
                    Accumulated_loss_50_batch = 0
        scheduler.step()

        epochTrainSMAPELost = epochTrainSMAPELost / len(train_dataset)
        epochTrainL1Lost = epochTrainL1Lost / len(train_dataset)
        epochTrainL2Lost = epochTrainL2Lost / len(train_dataset)
        epochTrainSSIMLost = epochTrainSSIMLost / len(train_dataset)

        epochValSMAPELost = epochValSMAPELost / len(test_dataset)
        epochValL1Lost = epochValL1Lost / len(test_dataset)
        epochValL2Lost = epochValL2Lost / len(test_dataset)
        epochValSSIMLost = epochValSSIMLost / len(test_dataset)
            
        epoch_smape.append(epochTrainSMAPELost)
        epoch_l1.append(epochTrainL1Lost)
        epoch_l2.append(epochTrainL2Lost)
        epoch_ssim.append(epochTrainSSIMLost)

        epoch_Vsmape.append(epochValSMAPELost)
        epoch_Vl1.append(epochValL1Lost)
        epoch_Vl2.append(epochValL2Lost)
        epoch_Vssim.append(epochValSSIMLost)
        
        print (f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epochTrainSMAPELost:.8f}, Validation Loss: {epochValSMAPELost:.8f}, l1T: {epochTrainL1Lost:.2f}, l1V: {epochValL1Lost:.2f}, l2T: {epochTrainL2Lost:.2f}, l2V: {epochValL2Lost:.2f}, SSIMT: {1 - epochTrainSSIMLost:.4f}, SSIMV: {1 - epochValSSIMLost:.4f}')

        f.write(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epochTrainSMAPELost:.8f}, Validation Loss: {epochValSMAPELost:.8f}\n')
        if epochValSMAPELost < best_acc and phase == 'validation':
            torch.save(model.state_dict(), output_path)
            torch.save(optimizer.state_dict(), output_path + "optim.pth")
            print (f'Saving for Validation Loss: {epochValSMAPELost:.8f}')
            # best_model = copy.deepcopy(model.state_dict())
            best_acc = epochValSMAPELost
        
    # store 
    performace = {"epoch_smape" : epoch_smape, "epoch_l1":epoch_l1, "epoch_l2":epoch_l2, "epoch_ssim":epoch_ssim, "epoch_Vsmape":epoch_Vsmape,"epoch_Vl1":epoch_Vl1, "epoch_Vl2":epoch_Vl2, "epoch_Vssim":epoch_Vssim }
    performace["savedModelValidation"] = best_acc
    # torch.save(best_model.state_dict(), output_path)
    with open(log_path+".json", 'w') as outfile:
        json.dump(performace, outfile)
    f.write('Ending at: %s\n' %datetime.now())
    f.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    output_path = "D:/1Full1NewDBmodel.pth"
    model = denoiseNet(final_layer_size=1)
    # model.load_state_dict(torch.load("D:/11FullNewDBmodel.pth"))
    model.to(device)
    dataset = DenoiseDataset("D:/274 New Training Dataset")

    training_detail = {}
    training_detail['num_epochs'] = 75
    training_detail['batch_size'] = 4 
    criterion = smapeLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)
    #optimizer.load_state_dict(torch.load("D:/11FullNewDBmodel.pthoptim.pth"))
    
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.75)
    train_model(model, criterion, optimizer, step_lr_scheduler, training_detail, dataset, output_path, 'C:/Users/Alsto/OneDrive - UC San Diego/CSE 274/training log/1Full1NewDBmodel',device)    



if __name__=="__main__":
    main()