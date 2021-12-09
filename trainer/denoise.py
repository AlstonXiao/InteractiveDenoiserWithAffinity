import torch
from torch import random
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from model import denoiseNet
from dataset import DenoiseDataset
from metrics import smape_np, l1_np
import cv2
import time
from utils import model_to_half
from piqa import psnr
# from pytorch_msssim import ssim
from skimage.metrics import structural_similarity as ssim
from torch.profiler import profile, record_function, ProfilerActivity
import sys

def denoise_image(model, device, imagePath, outputPath, spp=8):
    reference =cv2.imread(os.path.join(imagePath, "_reference.hdr"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED)
    reference_no_denoise =cv2.imread(os.path.join(imagePath, "_referenceNODenoise.hdr"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED)

    radiance = list()
    imageFloatSamples = list()
    imageboolSamples = list()
    for i in range(spp):
        radiancePS = cv2.imread(os.path.join(imagePath , str(i) + "_finalImage.hdr"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED)
        radiance.append(radiancePS.transpose((2, 0, 1)))

        diffusePS = cv2.imread(os.path.join(imagePath , str(i) + "_diffuse.hdr"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED)
        diffusePS = diffusePS.transpose((2, 0, 1))
        diffusePS = np.log(diffusePS + 1)
        specularPS = cv2.imread(os.path.join(imagePath , str(i) + "_specular.hdr"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED)
        specularPS = specularPS.transpose((2, 0, 1))
        specularPS = np.log(specularPS + 1)
        normalPS = cv2.imread(os.path.join(imagePath , str(i) + "_normal.hdr"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED)
        normalPS = normalPS.transpose((2, 0, 1))
        albedoPS = cv2.imread(os.path.join(imagePath , str(i) + "_albedo.hdr"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED)
        albedoPS = albedoPS.transpose((2, 0, 1))
        depthPS = cv2.imread(os.path.join(imagePath , str(i) + "_depth.hdr"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED)
        depthPS = depthPS.transpose((2, 0, 1))[0:1]
        roughPS = cv2.imread(os.path.join(imagePath , str(i) + "_roughness.hdr"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED)
        roughPS = roughPS.transpose((2, 0, 1))[0:1]
        imageFloatSamples.append(np.concatenate((diffusePS, specularPS, normalPS, albedoPS, depthPS, roughPS), 0))

        metallicPS = cv2.imread(os.path.join(imagePath , str(i) + "_metallic.png"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED).astype(np.bool_)
        emissivePS = cv2.imread(os.path.join(imagePath , str(i) + "_emissive.png"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED).astype(np.bool_)
        specularReflectPS = cv2.imread(os.path.join(imagePath , str(i) + "_specularReflect.png"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED).astype(np.bool_)

        intPS = np.array([metallicPS,emissivePS, specularReflectPS])
        imageboolSamples.append(intPS)
    # radiance 3, h, w
    # features 8, 17, h, w
    radiance = np.mean(radiance, axis=0)
    imageFloatSamples = np.array(imageFloatSamples).astype(np.float32)
    imageboolSamples = np.array(imageboolSamples).astype(np.float32)
    features = np.concatenate([imageFloatSamples, imageboolSamples],1)

    _, h, w = radiance.shape

    # we do need to pay a penalty of 8 pixels per dimension
    h_number_iter = (h - 16 - 1) // 128 + 1
    w_number_iter = (w - 16 - 1) // 128 + 1
    finalImage = np.zeros((h, w, 3), dtype=np.float32)
    for height in range(h_number_iter):
        for width in range(w_number_iter):
            starting_height = min(height * 128, h-144)
            starting_width = min(width * 128, w-144)
            image_starting_height = starting_height + 8 
            image_starting_width = starting_width + 8
            image_feature = torch.from_numpy(features[:, :, starting_height:starting_height+144, starting_width:starting_width+144]).view(1, 8, 17, 144, 144)
            image_radiance = torch.from_numpy(radiance[:, starting_height + 3:starting_height+141, starting_width + 3:starting_width+141]).view(1, 3, 138, 138)
            
            unfold = torch.nn.Unfold(kernel_size = (11, 11))
            image_radiance = (unfold(image_radiance)).view(1,363,128,128)

            image_feature = image_feature.to(device).half()
            image_radiance = image_radiance.to(device)

            modelInput = {"radiance" : image_radiance, "features" : image_feature}
            outputs,_ = model(modelInput)
            outputs = outputs.to('cpu').detach().numpy()[0].transpose(1,2,0)
            finalImage[image_starting_height:image_starting_height+128 , image_starting_width:image_starting_width+128, :] = outputs
    # cropping and clipping the image
    finalImage = finalImage[8:h-8, 8:w-8, :]
    reference = reference[8:h-8, 8:w-8, :]
    reference_no_denoise = reference_no_denoise[8:h-8, 8:w-8, :]

    # psnr
    psnr = cv2.PSNR(np.power(reference/(1+reference), 1/2.4), np.power(finalImage/(1+finalImage), 1/2.4), 1)
    print(f"PSNR: {psnr}")

    # ssim
    ssim_index = ssim(reference, finalImage, data_range=finalImage.max() - finalImage.min(), channel_axis=2)
    print(f"ssim: {1-ssim_index}")

    # smape
    smape_score = smape_np(finalImage, reference)
    print(f"smape_score: {smape_score}")

    l1_score = l1_np(finalImage, reference)
    print(f"l1_score: {l1_score}")

    # showcase image
    cv2.imwrite(outputPath,finalImage)
    finalImage = np.flip(finalImage, 0)
    reference = np.flip(reference, 0)
    cv2.imshow("denoised Image",finalImage)
    cv2.imshow("reference Image",reference)
    cv2.waitKey(0)

def denoise_training_set(model, training_dataset_path, sampleIndex, device, noisy_ref = False):
    sample = np.load(os.path.join(training_dataset_path, str(sampleIndex) + "_traningSample.npy"), allow_pickle=True)
    ref = sample[0][:,8:136, 8:136].transpose(1,2,0)
    if (noisy_ref):
        ref = sample[4][:,8:136, 8:136].transpose(1,2,0)

    rad = torch.from_numpy(sample[1]).view(1,3,144,144)
    rad = rad[:,:,3:141, 3:141]

    unfold = torch.nn.Unfold(kernel_size = (11, 11))
    rad = (unfold(rad)).view(1,363,128,128)
    
    floats = sample[2].astype(np.float32)
    ints = sample[3].astype(np.float32)
    features = torch.from_numpy(np.concatenate([floats, ints],1)).view(1, 8, 17, 144, 144)

    rad = rad.to(device)
    features = features.to(device)#.half()
    rad1 = rad
    features1 = features

    # testing the maximum size of a group
    for i in range(10):
        rad1 = torch.cat([rad1, rad])
        features1= torch.cat([features1, features])
    modelInput = {"radiance" : rad1, "features" : features1}
    
    s = time.time()
    # warm up the cache
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        with record_function("model_inference"):
            outputs,kernel = model(modelInput)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        with record_function("model_inference"):
            outputs,kernel = model(modelInput)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        with record_function("model_inference"):
            outputs,kernel = model(modelInput)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        with record_function("model_inference"):
            outputs,kernel = model(modelInput)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        with record_function("model_inference"):
            outputs,kernel = model(modelInput)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    time_taken = (time.time()-s )*1000
    print(f"Time taken: {time_taken}")

    outputs = outputs.to('cpu')
    kernel  = kernel.to('cpu')
    outputs = outputs.detach().numpy().astype(np.float32)[0].transpose(1,2,0)
    kernel  = kernel.detach().numpy().astype(np.float32)[0].transpose(1,2,0)

    # psnr
    psnr = cv2.PSNR(np.power(ref/(1+ref), 1/2.4), np.power(outputs/(1+outputs), 1/2.4), 1)
    print(f"PSNR: {psnr}")
    # ssim
    ssim_index = ssim(ref, outputs, data_range=outputs.max() - outputs.min(), channel_axis=2)
    print(f"ssim: {1-ssim_index}")

    # smape
    smape_score = smape_np(ref, outputs)
    print(f"smape_score: {smape_score}")

    cv2.imshow("reference", cv2.resize(ref, (512, 512), fx=0, fy=0, interpolation = cv2.INTER_NEAREST))
    radiance = sample[1][:,8:136, 8:136]
    cv2.imshow("before denoise", cv2.resize(radiance.astype(np.float32).transpose(1,2,0), (512, 512), fx=0, fy=0, interpolation = cv2.INTER_NEAREST))
    normal = sample[2][1, 6:9,8:136, 8:136]
    cv2.imshow("before denoise normal", cv2.resize(normal.astype(np.float32).transpose(1,2,0), (512, 512), fx=0, fy=0, interpolation = cv2.INTER_NEAREST))
    albedo = sample[2][1, 9:12,8:136, 8:136]
    cv2.imshow("before denoise albedo", cv2.resize(albedo.astype(np.float32).transpose(1,2,0), (512, 512), fx=0, fy=0, interpolation = cv2.INTER_NEAREST))

    cv2.imshow("after denoise",cv2.resize(outputs, (512, 512), fx=0, fy=0, interpolation = cv2.INTER_NEAREST))
    kernel = kernel*10
    kernelAtMiddle = kernel[50, 50].reshape(11,11)
    cv2.imshow("kernel",cv2.resize(kernelAtMiddle, (512, 512), fx=0, fy=0, interpolation = cv2.INTER_NEAREST))

    cv2.waitKey(0)

def generate_test_images():
    modelFile = "D:/11Full1NewDBmodel.pth"
    imageBasePath = "D:/274 Pure image reference/"
    outputBasePath = "D:/274 Pure image reference/"
    outfileName = "kernel11NoDe.hdr"
    inFileName = ["bathroom reference", "bedroom reference", "kitchen reference", "living room reference", "living-room2 reference",  "staircase reference" ]
    outFileName = ["bathroom", "bedroom", "kitchen", "living room", "living-room2",  "staircase" ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = denoiseNet(final_layer_size=11)
    model.load_state_dict(torch.load(modelFile, map_location=device))
    model.to(device)
    model_to_half(model)
    for i in range(6):
        denoise_image(model, device, imageBasePath+inFileName[i], outputBasePath+outFileName[i]+"/"+outfileName)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = denoiseNet(final_layer_size=int(sys.argv[2]))
    model.load_state_dict(torch.load(sys.argv[1], map_location=device))
    model.to(device)
    model_to_half(model)
    denoise_image(model, device, sys.argv[3], sys.argv[4])

    # denoise_training_set(model, "D:/274 New Training Dataset", 1085, device, True)