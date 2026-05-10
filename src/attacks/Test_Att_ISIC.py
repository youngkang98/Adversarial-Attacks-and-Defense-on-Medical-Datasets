# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:55:14 2024

@author: lkang
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

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
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
from math import floor
from utils.plot import make_adv_img
from utils.data import psnr
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
        plt.savefig(str(config.get_experiment_path(f'{name}.png')), dpi=300, bbox_inches='tight')
    plt.show()

# Define a mapping dictionary
label_mapping = {0: 0, 1: 1, 2: 2, 4: 3}
# label_mapping = {0: 0, 2: 1}

def remap_labels(labels):
    # Use the label_mapping dictionary to remap the labels
    return torch.tensor([label_mapping[label.item()] for label in labels])

# Update your evaluate function
def evaluate(classifier, loader, criterion, device, noise_tensor=None, add_noise=False, remap=False,eps_current=[0,0],image_count = 0):
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
                    make_adv_img(clean_image_for_psnr,noise_tensor,adv_img,str(config.get_experiment_path(f'adv_img_att{eps_current[0]}eps{eps_current[1]}_{image_count}.jpg')))
                    psnr(clean_image_np, adv_img_np,str(config.get_experiment_path(f'psnr_att{eps_current[0]}_eps{eps_current[1]}.txt')))
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

datapath = str(config.get_data_path('ISIC2019'))
# testfile = str(config.get_data_path('ISIC2019_test.csv'))
trainfile = str(config.get_data_path('ISIC2019_train_012.csv'))
testfile = str(config.get_data_path('ISIC2019_test_012.csv'))
train_data_path = str(config.get_data_path('ISIC2019'))
test_data_path = str(config.get_data_path('ISIC2019'))
adversarialFile = str(config.get_data_path('ISIC2019_Adversarial_012.csv'))

# Load the previously trained model
num_classes = 3
# checkpoint = torch.load(str(config.get_model_path('ISIC2019_morph.pth.tar')),map_location ='cpu')
# checkpoint = torch.load(str(config.get_model_path('ISIC2019_morph.pth.tar')),map_location ='cpu')
# checkpoint = torch.load(str(config.get_model_path('cifar10_morph.pth.tar')),map_location ='cpu')
# checkpoint = torch.load(str(config.get_model_path('ISIC2019_morph.pth.tar')),map_location ='cpu')

clean_checkpoint = torch.load(str(config.get_model_path('ISIC2019_morph.pth.tar')),map_location ='cpu')
checkpoint = torch.load(str(config.get_model_path('checkpoint.pth')),map_location ='cpu')

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

# Dataset for evaluation (excluding the images used for noise generation)
eval_dataset = ISICDataset(datapath, testfile, 'test_data', transform=test_transform, one_hot_encode= False, num_classes=num_classes)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True)
eval_dataset_length = len(eval_dataset)
print(f"Number of images in the evaluation dataset: {eval_dataset_length}")

adv_training_dataset = ISICDataset(datapath, trainfile, 'train_data', transform=test_transform, one_hot_encode= True, num_classes=num_classes)
adv_training_loader = DataLoader(adv_training_dataset, batch_size=16, shuffle=True)
adv_training_length = len(adv_training_dataset)
print(f"Number of images in the adv training dataset: {adv_training_length}")

# noise_dataset = ISICDataset(datapath, adversarialFile, 'train_data', transform=test_transform, one_hot_encode= False, num_classes=num_classes)
# noise_loader = DataLoader(noise_dataset, batch_size=16, shuffle=True)
# noise_length = len(noise_dataset)
# print(f"Number of images in the noise dataset: {noise_length}")

device = 'cuda'
model = resnet50()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define the loss function and the optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=5e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Create the ART classifier
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    input_shape=(3,256,256),
    optimizer=optimizer,
    nb_classes=num_classes,
    device_type='gpu',
    preprocessing=None
)

clean_model = resnet50()
clean_model.fc = nn.Linear(clean_model.fc.in_features, num_classes)
clean_model = clean_model.to(device)
clean_model.load_state_dict(clean_checkpoint['netC'])
clean_model.eval()


# Define the loss function and the optimizer
clean_criterion = torch.nn.CrossEntropyLoss()
clean_optimizer = torch.optim.SGD(clean_model.parameters(), 1e-4, momentum=0.9, weight_decay=5e-4)
clean_optimizer.load_state_dict(clean_checkpoint['optimizerC'])

# Create the ART classifier
clean_classifier = PyTorchClassifier(
    model=clean_model,
    loss=clean_criterion,
    input_shape=(3,256,256),
    optimizer=clean_optimizer,
    nb_classes=num_classes,
    device_type='gpu',
    preprocessing=None
)

# val_loss_no_noise, val_acc_no_noise, true_labels, pred_labels = evaluate(classifier, eval_loader, criterion, device,remap=remap)
# print(f"Without Noise after adv train - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%")

clean_x, clean_y = eval_dataset[0:]

#============================================================================
pgd = ProjectedGradientDescent(classifier, eps=1/255, eps_step=1/255, max_iter=10, num_random_init=0)
adv_x_pgd = pgd.generate(clean_x)

# Move the adversarial data to the appropriate device
adv_x_pgd = torch.tensor(adv_x_pgd).to(device)
clean_y = torch.tensor(clean_y).to(device)

# Use the ART classifier to predict
predictions = classifier.predict(adv_x_pgd.cpu().numpy())

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
correct = np.sum(predicted_labels == clean_y.cpu().numpy())
total = clean_y.size(0)
accuracy = correct / total

print(f'Accuracy on PGD adversarial examples: {accuracy * 100:.2f}%')

plot_confusion_matrix(clean_y.cpu().numpy(), predicted_labels)

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

uap = UniversalPerturbation(
    classifier,
    attacker='fgsm',
    attacker_params={'targeted': False, 'eps': 1/255}, #eps = 0.001,0.0024
    max_iter=15,
    eps= 1/255,#0.04
    norm=np.inf
)
adv_x_uap = uap.generate(clean_x)

# adversarial_dataset = AdversarialDataset(adv_x_fgsm, clean_y)
# adversarial_loader = DataLoader(adversarial_dataset, batch_size=32, shuffle=False)

# val_loss_no_noise, val_acc_no_noise, true_labels, pred_labels = evaluate(clean_classifier, eval_loader, criterion, device,remap=remap)
# print(f"Without Noise after adv train - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%")

# Move the adversarial data to the appropriate device
adv_x_uap = torch.tensor(adv_x_uap).to(device)
clean_y = torch.tensor(clean_y).to(device)

# Use the ART classifier to predict
predictions = classifier.predict(adv_x_uap.cpu().numpy())

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
correct = np.sum(predicted_labels == clean_y.cpu().numpy())
total = clean_y.size(0)
accuracy = correct / total

print(f'Accuracy on FGSM adversarial examples: {accuracy * 100:.2f}%')
plot_confusion_matrix(clean_y.cpu().numpy(), predicted_labels)

#====================================================================
# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Time taken to run the code: {elapsed_time} seconds")