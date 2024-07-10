# Standard library imports
import os
import time
import json
import datetime
import argparse
import yaml
import random 

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models_multiple_GPUs import *

# Set subdomains distribution  and other training parameters
subdomains_dist = (4,4)
depth = 4
complexity = 32
kernel_size = 5
padding = kernel_size // 2
comm = True
num_comm_fmaps = 256
num_epochs = 10000
exchange_fmaps = True

save_path = f"./results_new/baseline_{subdomains_dist[0]}x{subdomains_dist[1]}_{'with_comm' if comm else 'without_comm'}/"

# hyperparams
batch_size = 2
batch_size_test = 1

save_descr = f"{'with_comm' if comm else 'without_comm'}_{subdomains_dist[0]}"

class DatasetMultipleGPUs(Dataset):
    def __init__(self, image_labels, image_dir, mask_dir, transform=None, target_transform=None, 
                 data_augmentation=None, size=2560, patch_size = 1280, subdomains_dist=subdomains_dist):
        self.img_labels = image_labels
        self.img_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation
        self.size = size
        self.subdomains_dist = subdomains_dist
        self.patch_size = patch_size

    def __len__(self):
        return len(self.img_labels)
    
    def __split_image(self, full_image):
        subdomain_tensors = []
        subdomain_height = full_image.shape[2] // self.subdomains_dist[0]
        subdomain_width = full_image.shape[1] // self.subdomains_dist[1]

        for i in range(self.subdomains_dist[0]):
            for j in range(self.subdomains_dist[1]):
                subdomain = full_image[:, j * subdomain_height: (j + 1) * subdomain_height,
                            i * subdomain_width: (i + 1) * subdomain_width]
                subdomain_tensors.append(subdomain)

        return subdomain_tensors        
    
    def __crop_patch(self, full_image, full_mask):
        _, height, width = full_image.shape
        patch_height, patch_width = self.patch_size, self.patch_size

        if height < patch_height or width < patch_width:
            raise ValueError("Patch size must be smaller than image size.")
        
        top = random.randint(0, height - patch_height)
        top = 0
        left = random.randint(0, width - patch_width)
        left = 0
        
        image_patch = full_image[:, top:top + patch_height, left:left + patch_width]
        mask_patch = full_mask[:, top:top + patch_height, left:left + patch_width]

        return image_patch, mask_patch

    def __getitem__(self, idx):
        img_name = self.img_labels[idx]
        
        img_path =  os.path.join(self.img_dir, f"{img_name}")                
        mask_path =  os.path.join(self.mask_dir, f"{img_name}")

        image = torch.load(img_path)
        mask = torch.load(mask_path)

        images = []

        image, mask = self.__crop_patch(image, mask)

        if self.data_augmentation:
            image, mask = self.data_augmentation(image, mask)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
            
        images = self.__split_image(image)      


        return images, mask

# SET DATALOADERS
train_dataset = DatasetMultipleGPUs(image_labels=[f"RUN_{i}.pt" for i in range(6)], image_dir="./data/Inputs/", mask_dir="./data/Labels/", transform=None,
                                    target_transform=None, data_augmentation=None, size=2560, subdomains_dist=subdomains_dist)

val_dataset = DatasetMultipleGPUs(image_labels=["RUN_6.pt"], image_dir="./data/Inputs/", mask_dir="./data/Labels/", transform=None,
                                    target_transform=None, data_augmentation=None, size=2560, subdomains_dist=subdomains_dist)

test_dataset = DatasetMultipleGPUs(image_labels=["RUN_7.pt"], image_dir="./data/Inputs/", mask_dir="./data/Labels/", transform=None,
                                    target_transform=None, data_augmentation=None, size=2560, subdomains_dist=subdomains_dist)

# Define dataloaders
dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #, num_workers=6)
dataloader_val = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False)# , num_workers=6)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)# , num_workers=6)

# Set devices
devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
print("Available GPUs:", devices)

# Save path
os.makedirs(save_path, exist_ok=True)



def plot_results(unet, savepath, epoch_number):
    def plot_subplot(position, image, title='', vmin=None, vmax=None):
        plt.subplot(4, 3, position)
        plt.axis("off")
        plt.imshow(image, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        # plt.colorbar(shrink=0.5)
        if title:
            plt.title(title)

    def process_and_plot(images, masks, start_pos):
        unet.eval()
        with torch.no_grad():
            predictions = unet([img.unsqueeze(0) for img in images]).cpu()
            full_images = unet.concatenate_tensors([img.unsqueeze(0) for img in images]).squeeze().cpu()

        for i in range(3):
            plot_subplot(start_pos + i, full_images[i].cpu())
            
        
        plot_subplot(start_pos + 3, predictions[0, 0].cpu(), vmin=0, vmax=1, title='Prediction')
        plot_subplot(start_pos + 4, masks.cpu()[0], vmin=0, vmax=1, title='Ground Truth')
        plot_subplot(start_pos + 5, torch.abs(masks.cpu()[0] - predictions[0, 0].cpu()), vmin=0, vmax=1, title='Error')

    plt.figure(figsize=(9, 12.5))
    
    # Adjust spacing between plots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    train_image, train_mask = train_dataset[0]
    process_and_plot(train_image, train_mask, 1)

    val_image, val_mask = val_dataset[0]
    process_and_plot(val_image, val_mask, 7)

    os.makedirs(os.path.join(savepath, "figures"), exist_ok=True)
    plt.savefig(os.path.join(savepath, "figures", f"epoch_{epoch_number}.png"), bbox_inches='tight')
    plt.close()

unet = MultiGPU_UNet_with_comm(n_channels=3, n_classes=1, input_shape=(2560, 2560), num_comm_fmaps=num_comm_fmaps, devices=devices, depth=depth,
                                   subdom_dist=subdomains_dist, bilinear=False, comm=comm, complexity=complexity, dropout_rate=0.0, kernel_size=kernel_size, 
                                   padding=padding, communicator_type=None, comm_network_but_no_communication=(not exchange_fmaps), 
                                   communication_network_def=CNNCommunicatorDilated)
unet.load_weights(load_path=os.path.join(save_path, "unet.pth"), device=devices[0])
unet.eval()

plot_results(unet=unet, savepath="./", epoch_number=save_descr)
