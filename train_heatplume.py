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
subdomains_dist = (1,1)
depth = 4
complexity = 32
save_path = "./results_new/baseline_1x1_with_comm/"
kernel_size = 5
padding = kernel_size // 2
comm = True
num_comm_fmaps = 256
num_epochs = 10000
exchange_fmaps = True

# Train hyperparams
batch_size = 2
batch_size_test = 1

def save_parameters_to_json():
    # Set subdomains distribution and other training parameters
    parameters = {
        "subdomains_dist": subdomains_dist,
        "depth": depth,
        "complexity": complexity,
        "save_path": save_path,
        "kernel_size": kernel_size,
        "padding": padding,
        "comm": comm,
        "num_comm_fmaps": num_comm_fmaps,
        "num_epochs": num_epochs,
        "exchange_fmaps": exchange_fmaps,
        "train_hyperparams": {
            "batch_size": batch_size,
            "batch_size_test": batch_size_test
        }
    }

    # Ensure the save_path directory exists
    if not os.path.exists(parameters["save_path"]):
        os.makedirs(parameters["save_path"])

    # Define the file path
    file_path = os.path.join(parameters["save_path"], 'parameters.json')

    # Save the parameters to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(parameters, json_file, indent=4)

    print(f"Parameters saved to {file_path}")

    return parameters

parameters = save_parameters_to_json()
print(parameters)

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
        left = random.randint(0, width - patch_width)
        
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

# Define a data augmentation function
def data_augmentation(image, mask):
    # Random horizontal flip
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, [2])
        mask = torch.flip(mask, [2])

    # Random vertical flip
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, [1])
        mask = torch.flip(mask, [1])

    # Random rotation by 0, 90, 180, or 270 degrees
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        image = torch.rot90(image, k, [1, 2])
        mask = torch.rot90(mask, k, [1, 2])

    return image, mask


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

# Function to compute validation loss
def compute_validation_loss(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = [im.float() for im in images]
            masks = masks.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, masks)
            total_loss += loss.item()
            num_batches += 1
    
    average_loss = total_loss / num_batches
    return average_loss

def plot_results(unet, savepath, epoch_number):
    def plot_subplot(position, image, title='', vmin=None, vmax=None):
        plt.subplot(4, 3, position)
        plt.axis("off")
        plt.imshow(image, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        # plt.colorbar(shrink=0.5)
        # if title:
        #     plt.title(title)

    def process_and_plot(images, masks, start_pos):
        unet.eval()
        with torch.no_grad():
            predictions = unet([img.unsqueeze(0) for img in images]).cpu()
            full_images = unet.concatenate_tensors([img.unsqueeze(0) for img in images]).squeeze().cpu()

        for i in range(3):
            plot_subplot(start_pos + i, full_images[i].cpu())
        
        plot_subplot(start_pos + 3, predictions[0, 0].cpu(), vmin=0, vmax=1)
        plot_subplot(start_pos + 4, masks.cpu()[0])
        plot_subplot(start_pos + 5, torch.abs(masks.cpu()[0] - predictions[0, 0].cpu()), vmin=0, vmax=1)

    plt.figure(figsize=(9, 12))
    
    # Adjust spacing between plots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    train_image, train_mask = train_dataset[0]
    process_and_plot(train_image, train_mask, 1)

    val_image, val_mask = val_dataset[0]
    process_and_plot(val_image, val_mask, 7)

    os.makedirs(os.path.join(savepath, "figures"), exist_ok=True)
    plt.savefig(os.path.join(savepath, "figures", f"epoch_{epoch_number}.png"), bbox_inches='tight')
    plt.close()

# Function to train parallel model
def test_parallel_model(comm=True, num_epochs=25, num_comm_fmaps=64, save_path=save_path, subdomain_dist=subdomains_dist, exchange_fmaps=False):
    
    if num_comm_fmaps == 0:
        comm = False
    
    unet = MultiGPU_UNet_with_comm(n_channels=3, n_classes=1, input_shape=(2560, 2560), num_comm_fmaps=num_comm_fmaps, devices=devices, depth=depth,
                                   subdom_dist=subdomain_dist, bilinear=False, comm=comm, complexity=complexity, dropout_rate=0.0, kernel_size=kernel_size, 
                                   padding=padding, communicator_type=None, comm_network_but_no_communication=(not exchange_fmaps), 
                                   communication_network_def=CNNCommunicatorDilated)
    
    unet.save_weights(save_path=os.path.join(save_path, "unet.pth"))
              
    if comm:
        parameters = list(unet.encoders[0].parameters()) + list(unet.decoders[0].parameters()) + list(unet.communication_network.parameters()) 
    else:
        parameters = list(unet.encoders[0].parameters()) + list(unet.decoders[0].parameters())
        
    optimizer = torch.optim.Adam(parameters, lr=5e-5, weight_decay=1e-4)
    loss = torch.nn.MSELoss() #DiceLoss(mode="multiclass")
    losses = []

    # Wrap your training loop with tqdm
    start_time = time.time()
    validation_losses = []
    best_val_loss = float('inf')

    # Iterate over the epochs
    for epoch in range(num_epochs):
        unet.train()
        epoch_losses = []  # Initialize losses for the epoch
        
        for images, masks in tqdm(dataloader_train):
            optimizer.zero_grad()
            
            # Data loading and sending to the correct device
            images = [im.float() for im in images]
            masks = masks.to(devices[0])

            ## Forward propagation:
            # Run batch through encoder
            predictions = unet(images)
            

            ## Backward propagation
            l = loss(predictions, masks)
            l.backward()
            
            losses.append(l.item())  # Append loss to global losses list
            epoch_losses.append(l.item())  # Append loss to epoch losses list

            with torch.no_grad():
                for i in range(1, len(unet.encoders)):
                    for param1, param2 in zip(unet.encoders[0].parameters(), unet.encoders[i].parameters()):
                        if param1.grad is not None:
                            param1.grad += param2.grad.to(devices[0])
                            param2.grad = None

            with torch.no_grad():
                for i in range(1, len(unet.decoders)):
                    for param1, param2 in zip(unet.decoders[0].parameters(), unet.decoders[i].parameters()):
                        param1.grad += param2.grad.to(devices[0])
                        param2.grad = None
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            for i in range(1, len(unet.encoders)):
                unet.encoders[i].load_state_dict(unet.encoders[0].state_dict())
                
            for i in range(1, len(unet.decoders)):
                unet.decoders[i].load_state_dict(unet.decoders[0].state_dict())
    
        # Compute and print validation loss
        val_loss = compute_validation_loss(unet, loss, dataloader_val, devices[0])
        print(f'Validation Loss (Dice): {val_loss:.4f}, Train Loss: {losses[-1]:.4f}')
        
        validation_losses.append(val_loss)
        
        # Check for improvement and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            unet.save_weights(save_path=os.path.join(save_path, "unet.pth"))
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if (epoch + 1) % 50 == 0:
            plot_results(unet=unet, savepath=save_path, epoch_number=epoch)

        # if torch.cuda.is_available()
        #     # Track maximum GPU memory used
        #     max_memory_used = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        #     print(f"Maximum GPU Memory Used in Epoch {epoch+1}: {max_memory_used:.2f} GB")
        #     torch.cuda.reset_peak_memory_stats()

    print(f"Training the model {'with' if comm else 'without'} communication network took: {time.time() - start_time:.2f} seconds.")
    
    # Load the best weights
    unet.load_weights(load_path=os.path.join(save_path, "unet.pth"), device=devices[0])
    
    return unet, validation_losses, losses

unet, losses, training_losses = test_parallel_model(comm=comm, num_comm_fmaps=num_comm_fmaps, num_epochs=num_epochs, exchange_fmaps=exchange_fmaps)

plot_results(unet=unet, savepath=save_path, epoch_number="best")

# Plot and save the losses
plt.plot(losses)
plt.plot(training_losses)
plt.savefig(f"{save_path}/loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# After the training loop, save the losses to a JSON file
with open(f"{save_path}/losses.json", "w") as f:
    json.dump(losses, f)
    json.dump(training_losses, f)