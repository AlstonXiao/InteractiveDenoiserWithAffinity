import os
import struct
import torch
import random
import cv2
import numpy as np
from torch.utils.data import Dataset

def createDataset(path, outputPath, spp, cropPerImage):
    bigImages = os.listdir(path)
    random.seed(0)
    count = 0
    for image in bigImages:
        imagePath = os.path.join(path, image)
        reference =cv2.imread(os.path.join(imagePath, "_reference.hdr"), flags=cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED)
        reference = reference.transpose((2, 0, 1))

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

        radiance = np.mean(radiance, axis=0)
        imageFloatSamples = np.array(imageFloatSamples)
        imageboolSamples = np.array(imageboolSamples)
        for i in range(cropPerImage):
            cropX = random.randrange(0, 1024-144)
            cropY = random.randrange(0, 1024-144)
            rotation = random.randrange(4)
            HFlip = random.randrange(2)
            VFlip = random.randrange(2)
            ref = np.array(reference[:,cropX:cropX+144,cropY:cropY+144]).astype(np.float16)
            rad= np.array(radiance[:,cropX:cropX+144,cropY:cropY+144]).astype(np.float16)
            fltPS = np.array(imageFloatSamples[:,:,cropX:cropX+144,cropY:cropY+144]).astype(np.float16)
            boolPS = np.array(imageboolSamples[:,:,cropX:cropX+144,cropY:cropY+144])
            for j in range(rotation):
                ref = np.rot90(ref, axes=(1,2))
                rad = np.rot90(rad, axes=(1,2))
                fltPS = np.rot90(fltPS, axes=(2,3))
                boolPS = np.rot90(boolPS, axes=(2,3))
            if HFlip > 0:
                ref = np.flip(ref, 1)
                rad = np.flip(rad, 1)
                fltPS = np.flip(fltPS, 2)
                boolPS = np.flip(boolPS, 2)
            if VFlip > 0:
                ref = np.flip(ref, 2)
                rad = np.flip(rad, 2)
                fltPS = np.flip(fltPS, 3)
                boolPS = np.flip(boolPS, 3)
            # cv2.imshow("we",fltPS[0][6:9].transpose(1,2,0).astype(np.float32))
            # cv2.waitKey(0)
            np.save(os.path.join(outputPath, str(count)+"_traningSample.npy"), np.array([ref, rad, fltPS, boolPS]))
            count += 1

if __name__ == '__main__':
    createDataset("C:/Users/Alsto/OneDrive - UC San Diego/CSE 274/dataset", "D:/274 Traning Dataset",8, 25)