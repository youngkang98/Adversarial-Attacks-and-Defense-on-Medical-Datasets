# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:35:41 2024

@author: lkang
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from dataloader import ISICDataset, get_transform
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import UniversalPerturbation,TargetedUniversalPerturbation
from art.defences.trainer import AdversarialTrainerFBFPyTorch,AdversarialTrainerMadryPGD
from art.data_generators import PyTorchDataGenerator
from art.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
from math import floor
from utils.plot import make_adv_img
from utils.data import psnr
import os
# from utils.test import test

from art.defences.trainer import AdversarialTrainer
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent,DeepFool
from PreActBottleNeck import PreActResNet50

import torchvision.datasets as datasets

import warnings
warnings.filterwarnings("ignore")

import time

# Start the timer
start_time = time.time()

def plot_confusion_matrix(true_labels,pred_labels,name=None):
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Display the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if name != None:
        # Save the figure
        plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Define a mapping dictionary
label_mapping = {0: 0, 1: 1, 2: 2, 4: 3}
# label_mapping = {0: 0, 2: 1}

def remap_labels(labels):
    # Use the label_mapping dictionary to remap the labels
    return torch.tensor([label_mapping[label.item()] for label in labels])

# Update your evaluate function
def evaluate_2(classifier, loader, criterion, device, noise_tensor=None, add_noise=False, remap=False,eps_current=[0,0],image_count = 0):
    classifier._model.eval()
    val_loss = 1.0
    val_acc = 0.0
    true_labels = []
    pred_labels = []
    first_image = True

    with torch.no_grad():
        for images, labels in loader:
            if remap:
                images, labels = images.to(device), remap_labels(labels).to(device)  # Remap labels here
            else:
                images, labels = images.to(device), labels.to(device)
            true_labels.extend(labels.cpu().numpy())
            
            if add_noise and noise_tensor is not None:
                clean_image_for_psnr = images[0].clone()
                images += noise_tensor
                adv_img = images[0]
                clean_image_np = np.transpose(clean_image_for_psnr.cpu().numpy(), (1, 2, 0))
                adv_img_np = np.transpose(adv_img.cpu().numpy(), (1, 2, 0))
                
                if first_image:
                    make_adv_img(clean_image_for_psnr,noise_tensor,adv_img,f'ISIC2019/adv_img_att{eps_current[0]}eps{eps_current[1]}_{image_count}.jpg')
                    psnr(clean_image_np, adv_img_np,f'ISIC2019/psnr_att{eps_current[0]}_eps{eps_current[1]}.txt')
                    first_image = False

            outputs = classifier.predict(images.cpu().numpy())
            outputs = torch.tensor(outputs).to(device)
            _, preds = torch.max(outputs, 1)
            pred_labels.extend(preds.cpu().numpy())
            val_acc += torch.sum(preds == labels.data)
            loss = criterion(outputs, labels)  # Now the labels are in the correct range
            val_loss += loss.item() * images.size(0)

    val_loss /= len(loader.dataset)
    val_acc = val_acc.double() / len(loader.dataset)
    return val_loss, val_acc * 100, true_labels, pred_labels  # Return true_labels and pred_labels


def save_results_to_file(filename, val_loss_no_noise, val_acc_no_noise, val_loss_with_noise, val_acc_with_noise, success_rate = 0, targeted = False):
    if(not targeted):
        with open(filename, 'w') as file:
            file.write(f"Without Noise - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%\n")
            file.write(f"With Noise - Val Loss: {val_loss_with_noise:.4f} - Val Acc: {val_acc_with_noise:.2f}%\n")
    else:
        with open(filename, 'w') as file:
            file.write(f"Without Noise - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%\n")
            file.write(f"With Noise - Val Loss: {val_loss_with_noise:.4f} - Val Acc: {val_acc_with_noise:.2f}% - Succ Rate: {success_rate:.2f}%")

def to_one_hot(label, num_classes):
    one_hot_vector = np.zeros(num_classes)
    one_hot_vector[label] = 1
    return one_hot_vector

# After the evaluate function or where true_labels and pred_labels are available
def generate_classification_report(true_labels, pred_labels,fileName):
    # Generate the classification report
    report = classification_report(true_labels, pred_labels)
    print(report)

    # Save the report to a text file
    with open(fileName, 'w') as f:
        f.write(report)
        
class AdversarialDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data).float()  # Ensure data is float
        self.labels = torch.tensor(labels).long()  # Ensure labels are long integers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

        
# -------------------------------------------------------
# Parameters need to change

datapath = 'C:/Users/lkang/Documents/ISIC_2019_Training_Input/'
# testfile = 'C:/Users/lkang/Documents/New UAP/ISIC2019_test.csv'
trainfile = 'C:/Users/lkang/Documents/New UAP/ISIC2019_train_012.csv'
testfile = 'C:/Users/lkang/Documents/New UAP/ISIC2019_test_012.csv'
# train_data_path = 'C:/Users/lkang/Documents/ISIC_2019_train/'
# test_data_path = 'C:/Users/lkang/Documents/ISIC_2019_test/'
train_data_path = '../chest_xray/train'
test_data_path = '../chest_xray/test'

num_classes = 4

clean_checkpoint = torch.load('C:/Users/lkang/Documents/ISIC_Model/Acc81_Epoch100_BS16_3class/ISIC2019_morph.pth.tar',map_location ='cpu')
checkpoint = torch.load('C:/Users/lkang/Documents/ISIC_Model/Acc70_advtrain_epoch100_BS32_3class/checkpoint.pth',map_location ='cpu')

# Number of images to use for noise generation
image_count = 1773

# List of numbers of images to use for noise 
image_counts = [1773]

# iterations = [35,30,25,20,15,10]
iterations = [25]
# eps = [0.0005,0.001,0.002,0.005,0.01,0.02,0.03,0.04,0.05]
# attack_eps = [0.0005,0.001,0.002,0.005,0.01,0.02,0.03,0.04,0.05]
eps = [0.04]
attack_eps = [0.0024]
remap = False
targeted_attack = False

# ----------------------------------------------------------

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset =  datasets.ImageFolder(train_data_path,train_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_dataset_length = int(len(train_dataset))
print(f"Number of images in the train dataset: {train_dataset_length}")

eval_dataset = datasets.ImageFolder(test_data_path,test_transform)
eval_dataset_length = len(eval_dataset)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True)
print(f"Number of images in the evaluation dataset: {eval_dataset_length}")

device = 'cuda'
# Load the model
model = resnet50()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
model.train()
# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

# # Create the ART classifier
# classifier = PyTorchClassifier(
#     model=model,
#     loss=criterion,
#     input_shape=(3, 256, 256),
#     optimizer=optimizer,
#     nb_classes=num_classes,
#     device_type='gpu'
# )

# Training function
def train(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}")

# Evaluation function
def evaluate(model, eval_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return running_loss / len(eval_loader), accuracy

# Training loop
num_epochs = 50
train(model, train_loader, criterion, optimizer, device, num_epochs)

# Evaluate the model
eval_loss, eval_accuracy = evaluate(model, eval_loader, criterion, device)
print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.2f}%")


save_path = 'model'
# # Save the classifier, model, and optimizer state
# checkpoint = {
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'classifier': classifier,
# }

# os.makedirs(save_path, exist_ok=True)
# torch.save(checkpoint, os.path.join(save_path, 'OCT_Model_epoch50_BS16.pth'))

# Save the model and optimizer state
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}

os.makedirs(save_path, exist_ok=True)
torch.save(checkpoint, os.path.join(save_path, 'chest_xray_epoch50_BS16.pth'))
# val_loss_no_noise, val_acc_no_noise, true_labels, pred_labels = evaluate(classifier, eval_loader, criterion, device,remap=remap)
# print(f"Without Noise after adv train - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%")

# clean_x, clean_y = eval_dataset[0:]


#===========================================================================

# fgsm = FastGradientMethod(classifier, eps=1/255, eps_step=1/255, num_random_init=0)
# adv_x_fgsm = fgsm.generate(clean_x)

# # adversarial_dataset = AdversarialDataset(adv_x_fgsm, clean_y)
# # adversarial_loader = DataLoader(adversarial_dataset, batch_size=32, shuffle=False)

# # val_loss_no_noise, val_acc_no_noise, true_labels, pred_labels = evaluate(clean_classifier, eval_loader, criterion, device,remap=remap)
# # print(f"Without Noise after adv train - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%")

# # Move the adversarial data to the appropriate device
# adv_x_fgsm = torch.tensor(adv_x_fgsm).to(device)
# clean_y = torch.tensor(clean_y).to(device)

# # Use the ART classifier to predict
# predictions = classifier.predict(adv_x_fgsm.cpu().numpy())

# # Convert predictions to class labels
# predicted_labels = np.argmax(predictions, axis=1)

# # Calculate accuracy
# correct = np.sum(predicted_labels == clean_y.cpu().numpy())
# total = clean_y.size(0)
# accuracy = correct / total

# print(f'Accuracy on FGSM adversarial examples: {accuracy * 100:.2f}%')
# plot_confusion_matrix(clean_y.cpu().numpy(), predicted_labels)

#=======================================================================

# deep_fool = DeepFool(clean_classifier, epsilon=0.004, max_iter=10)
# adv_x_df = deep_fool.generate(clean_x)

# # Move the adversarial data to the appropriate device
# adv_x_df = torch.tensor(adv_x_df).to(device)
# clean_y = torch.tensor(clean_y).to(device)

# # Use the ART classifier to predict
# predictions = classifier.predict(adv_x_df.cpu().numpy())

# # Convert predictions to class labels
# predicted_labels = np.argmax(predictions, axis=1)

# # Calculate accuracy
# correct = np.sum(predicted_labels == clean_y.cpu().numpy())
# total = clean_y.size(0)
# accuracy = correct / total

# print(f'Accuracy on Deep Fool adversarial examples: {accuracy * 100:.2f}%')
# plot_confusion_matrix(clean_y.cpu().numpy(), predicted_labels)


#===================================================================

# uap = UniversalPerturbation(
#     classifier,
#     attacker='fgsm',
#     attacker_params={'targeted': False, 'eps': 1/255}, #eps = 0.001,0.0024
#     max_iter=15,
#     eps= 1/255,#0.04
#     norm=np.inf
# )
# adv_x_uap = uap.generate(clean_x)

# # adversarial_dataset = AdversarialDataset(adv_x_fgsm, clean_y)
# # adversarial_loader = DataLoader(adversarial_dataset, batch_size=32, shuffle=False)

# # val_loss_no_noise, val_acc_no_noise, true_labels, pred_labels = evaluate(clean_classifier, eval_loader, criterion, device,remap=remap)
# # print(f"Without Noise after adv train - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%")

# # Move the adversarial data to the appropriate device
# adv_x_uap = torch.tensor(adv_x_uap).to(device)
# clean_y = torch.tensor(clean_y).to(device)

# # Use the ART classifier to predict
# predictions = classifier.predict(adv_x_uap.cpu().numpy())

# # Convert predictions to class labels
# predicted_labels = np.argmax(predictions, axis=1)

# # Calculate accuracy
# correct = np.sum(predicted_labels == clean_y.cpu().numpy())
# total = clean_y.size(0)
# accuracy = correct / total

# print(f'Accuracy on FGSM adversarial examples: {accuracy * 100:.2f}%')
# plot_confusion_matrix(clean_y.cpu().numpy(), predicted_labels)

#====================================================================
# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Time taken to run the code: {elapsed_time} seconds")