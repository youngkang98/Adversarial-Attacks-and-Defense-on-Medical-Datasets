# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:48:41 2023

@author: lkang
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision import models
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm
from PIL import Image

def train_model(model, criterion, optimizer, num_epochs=3):
    since = time.time()
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        for mode, data_loader in data_loaders.items():
            if mode == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            # Iterate over batches
            for inputs, labels in tqdm(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(mode == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if mode == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds==labels.data)
            
            data_len = train_len
            
            if mode == 'val':
                data_len = val_len
            
            epoch_loss = running_loss / data_len
            epoch_acc = running_correct.double() / data_len

            print('{} loss: {:4f} acc: {:4f}'.format(mode, epoch_loss, epoch_acc))
        

class CustomImageDataset(Dataset):
    def __init__(self, ds_path, transforms=None, target_transform=None):
        self.ds_path = ds_path
        self.transforms = transforms
        self.target_transform = target_transform
        self.labels = os.listdir(ds_path)
        img_paths = []
        
        for i in self.labels:
            base_path = os.path.join(ds_path, i)
            imgs = os.listdir(base_path)
            for img in imgs:
                img_path = base_path + '\\' + img
                img_paths.append(img_path)
                
        self.img_paths = img_paths
        self.label_map = {'CNV':0, 'DME':1, 'DRUSEN':2, 'NORMAL':3,}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        # label = img_path.split('/')[-2]
        # label = img_path.split(os.path.sep)[-3]
        label = img_path.split('\\')[-2]  # Using backslash as separator
        image = image.squeeze()
        image = image.repeat(3, 1, 1)
        image = image / 255.0
        if self.transforms:
            image = self.transforms(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        image = T.Resize((224, 224))(image)
        
        return image, self.label_map[label]

# Define the OCTDataset class
class OCTDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.datapath = datapath
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Load data from folders
        self._load_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.datapath, self.data[idx])
        # image = Image.open(img_name).convert("L")
        image = read_image(img_name)
        image = image.squeeze()
        image = image.repeat(3, 1, 1)
        image = image / 255.0
        image = T.Resize((224, 224))(image)
        label = self.labels[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def _load_data(self):
        class_folders = [d for d in os.listdir(self.datapath) if os.path.isdir(os.path.join(self.datapath, d))]
        class_folders.sort()
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_folders)}
        
        for class_name in class_folders:
            class_dir = os.path.join(self.datapath, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    image_path = os.path.join(class_name, img_name)
                    self.data.append(image_path)
                    self.labels.append(class_to_idx[class_name])
    
    # Load training dataset
train_path = 'C:/Users/lkang/Documents/archive/OCT2017/train'
# train_path = 'C:\\Users\\lkang\\Documents\\archive\\OCT2017\\train'
# train_data = CustomImageDataset(train_path)
train_data = OCTDataset(train_path)

# Load validation dataset
val_path = 'C:/Users/lkang/Documents/archive/OCT2017/val'
# val_data = CustomImageDataset(val_path)
val_data = OCTDataset(val_path)

# Load test dataset
test_path = 'C:/Users/lkang/Documents/archive/OCT2017/test'
# test_data = CustomImageDataset(test_path)
test_data = OCTDataset(test_path)

from torch.utils.data import DataLoader

train_len = len(train_data)
train_dataloader = DataLoader(train_data, num_workers=8, batch_size=64, shuffle=True)

val_len = len(val_data)
val_dataloader = DataLoader(val_data,num_workers=8, batch_size=64, shuffle=True)

test_len = len(test_data)
test_dataloader = DataLoader(test_data, batch_size=8)

data_loaders = {'train':train_dataloader, 'val':val_dataloader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cuda"

# model = models.resnet50()
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 10)

model = models.resnet50()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001)

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # Adjust to match the number of classes in your checkpoint

for param in model.parameters():
    param.required_grad = False

checkpoint = torch.load('C:/Users/lkang/Documents/OCT_Acc25_Epoch50_BS16_Clean/OCT_morph.pth.tar',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['netC'])
optimizer.load_state_dict(checkpoint['optimizerC'])
# train_model(model, criterion, optimizer, num_epochs=10)
model.eval()


running_loss = 0.0
running_correct = 0

# Iterate over batches
for inputs, labels in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

    running_loss += loss.item() * inputs.size(0)
    running_correct += torch.sum(preds==labels.data)
    
epoch_loss = running_loss / test_len
epoch_acc = running_correct.double() / test_len

print('{} loss: {:4f} acc: {:4f}'.format('test', epoch_loss, epoch_acc))

PATH = 'OCT.pth.tar'
torch.save(model, PATH)

