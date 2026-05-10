# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:33:05 2023

@author: lkang
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet201
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from dataloader import ISICDataset, load_data

# Step 1: Load the ISIC dataset
datapath = str(config.get_data_path('ISIC2019'))
trainfile = str(config.get_data_path('ISIC2019_train.csv'))
testfile = str(config.get_data_path('ISIC2019_test.csv'))
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
train_dataset = ISICDataset(datapath, trainfile, 'train_data',transform=train_transform)
test_dataset = ISICDataset(datapath, testfile,'test_data', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Step 2: Create the DenseNet-201 model
model = densenet201(pretrained=True)
num_classes = 8
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, num_classes)

# Step 3: Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Step 4: Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the previously trained model
# model_path = 'path_to_your_saved_model/epoch10_model.pth'
# model.load_state_dict(torch.load(model_path))
# model = model.to(device)

def train(model, loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels.data)
    train_loss /= len(loader.dataset)
    train_acc = train_acc.double() / len(loader.dataset)
    return train_loss, train_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)
    val_loss /= len(loader.dataset)
    val_acc = val_acc.double() / len(loader.dataset)
    return val_loss, val_acc

num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), str(config.get_model_path('epoch10_model.pth')))