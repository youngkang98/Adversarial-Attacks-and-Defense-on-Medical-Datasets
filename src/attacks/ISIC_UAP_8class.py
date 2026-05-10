# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:13:13 2024

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
from torch.utils.data import DataLoader
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
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from PreActBottleNeck import PreActResNet50

import torchvision.datasets as datasets

import warnings
warnings.filterwarnings("ignore")

import time

# Start the timer
start_time = time.time()

def plot_confusion_matrix(true_labels,pred_labels,name):
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Display the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
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

def evaluate_targeted_attack(classifier, loader, target_class, criterion, device, noise_tensor=None, add_noise=False, remap = False,eps_current=[0,0],image_count = 0):
    classifier._model.eval()
    successful_attacks = 0
    total_samples = 0
    val_loss = 0.0
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
            
            # Add noise if add_noise is True
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
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            pred_labels.extend(preds.cpu().numpy())
            val_acc += torch.sum(preds == labels.data)

            total_samples += len(labels)
            # successful_attacks += torch.sum((preds == target_class).int()).item()
            # original, as long as the pred falls into targeted class consider succ
            # successful_attacks += torch.sum(preds == target_class).item()
            # modified, only calculate when changed the pred after add the noise
            successful_attacks += torch.sum((labels != target_class) & (preds == target_class)).item()

    val_loss /= total_samples
    val_acc = (val_acc.double() / len(loader.dataset)) * 100
    success_rate = (successful_attacks / total_samples) * 100  # Multiply by 100 for percentage
    return val_loss, val_acc ,success_rate, true_labels, pred_labels

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

# -------------------------------------------------------
# Parameters need to change

datapath = str(config.get_data_path('ISIC2019'))
train_data_path = str(config.get_data_path('ISIC2019'))
test_data_path = str(config.get_data_path('ISIC2019'))

trainfile = str(config.get_data_path('ISIC2019_train.csv'))
testfile = str(config.get_data_path('ISIC2019_test.csv'))
adversarialFile = str(config.get_data_path('ISIC2019_train.csv'))

# Load the previously trained model
num_classes = 8
# Number of images to use for noise generation
image_count = 1773

# List of numbers of images to use for noise 
image_counts = [525,473,420,368,315,263,210,158,105,53]

# checkpoint
# checkpoint = torch.load(str(config.get_model_path('checkpoint.pth')),map_location ='cpu')
checkpoint = torch.load(str(config.get_model_path('classifier_checkpoint.pth')),map_location ='cpu')

iterations = [25]
eps = [0.04]
attack_eps = [0.0024]
remap = False
targeted_attack = False
# Specify the target class as an integer (e.g., 2 for "Basal cell carcinoma")
target_class = 2
# Define the different percentages for adversarial training
adv_percentages = [0.1, 0.09, 0.08, 0.07,0.06,0.05,0.04,0.03,0.02,0.01]  # Adjust these values as needed
# adv_percentages = [0.07]

# ----------------------------------------------------------

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    
])

# Dataset for evaluation (excluding the images used for noise generation)
eval_dataset = ISICDataset(datapath, testfile, 'test_data', transform=test_transform, one_hot_encode= False, num_classes=num_classes)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True)
eval_dataset_length = len(eval_dataset)
print(f"Number of images in the evaluation dataset: {eval_dataset_length}")

noise_dataset = ISICDataset(datapath, adversarialFile, 'train_data', transform=test_transform, one_hot_encode= False, num_classes=num_classes)
noise_loader = DataLoader(noise_dataset, batch_size=16, shuffle=True)
noise_length = len(noise_dataset)
print(f"Number of images in the adv training dataset: {noise_length}")

device = 'cuda'
model = resnet50()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
model.load_state_dict(checkpoint['model_state'])
# model.load_state_dict(checkpoint['netC'])
model.eval()


# Define the loss function and the optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=5e-4)
optimizer.load_state_dict(checkpoint['optimizer_state'])
# optimizer.load_state_dict(checkpoint['optimizerC'])

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

# Universal Perturbation parameters
mean_inf_train = 0.57  # Modify as needed

#========================

image_list = []

success_rate = 0

model.eval()

# Evaluation madry_pgd
val_loss_no_noise, val_acc_no_noise, true_labels, pred_labels = evaluate(classifier, eval_loader, criterion, device,remap=remap)
print(f"Without Noise after adv train - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%")
plot_confusion_matrix(true_labels, pred_labels, "confusion_matrix_clean_after_advtrain")
save_results_to_file(str(config.get_experiment_path('evaluation_result_after_adv_train.txt')),val_loss_no_noise,val_acc_no_noise, 0, 0, 0,targeted=False)
classificationReportFileName = str(config.get_experiment_path('classification_report_clean_after_advtrain.txt'))
generate_classification_report(true_labels,pred_labels,classificationReportFileName)


# Initialize data structures to hold the split data
all_images = []
all_labels = []

for images, labels in noise_loader:
    for img, label in zip(images, labels):
        all_images.append(img)
        all_labels.append(label)

# Initialize class-specific image collections
class_images = defaultdict(list)
for img, label in zip(all_images, all_labels):
    class_images[label.item()].append(img)

# Loop through the different numbers of images
for current_attack_eps in attack_eps:
    for current_eps in eps:
        for percentage in adv_percentages:
            # Calculate the number of images for adversarial and clean training based on the percentage
            adv_image_count = int(noise_length * percentage)  # Assuming total_image_count is defined
            clean_image_count = noise_length - adv_image_count
            
            # Initialize data structures to hold the split data
            adv_images = []
            adv_labels = []
            clean_images = []
            clean_labels = []

            # Split images into adversarial and clean training sets
            for class_idx, images in class_images.items():
                split_point = int(len(images) * percentage)
                random.shuffle(images)
                adv_images.extend(images[:split_point])
                adv_labels.extend([torch.tensor(class_idx)] * split_point)
                clean_images.extend(images[split_point:])
                clean_labels.extend([torch.tensor(class_idx)] * (len(images) - split_point))

            # Convert lists to tensors for training
            adv_x_subset = torch.stack(adv_images[:adv_image_count]).cpu().numpy()
            adv_y_subset = torch.tensor(adv_labels[:adv_image_count]).cpu().numpy()

            clean_x_subset = torch.stack(clean_images[:clean_image_count]).cpu().numpy()
            clean_y_subset = torch.tensor(clean_labels[:clean_image_count]).cpu().numpy()

            print(f"Adversarial training set class distribution: {defaultdict(int, {i: adv_labels.count(torch.tensor(i)) for i in range(num_classes)})}")
            print(f"Clean training set class distribution: {defaultdict(int, {i: clean_labels.count(torch.tensor(i)) for i in range(num_classes)})}")

            random.shuffle(adv_images)
            # Concatenate exactly 'adv_image_count' number of images for adversarial training
            x_subset = torch.stack(adv_images[:adv_image_count]).cpu().numpy()
            
            print(f"number of subset: {len(x_subset)}")
            if targeted_attack:
                # Generate targeted attack
                adv_crafter = TargetedUniversalPerturbation(
                    classifier,
                    attacker='fgsm',
                    delta=0.000001,
                    attacker_params={'targeted': True, 'eps': current_attack_eps},
                    max_iter=15,
                    eps=current_eps,
                    norm=np.inf
                )
                
                # Create a one-hot encoded array of target labels
                target_labels = np.array([to_one_hot(target_class, num_classes) for _ in range(len(x_subset))])
            
                # Generate adversarial examples with the target class
                x_test_adv = adv_crafter.generate(x=x_subset, y=target_labels)
            else:
                # Generate universal attack
                adv_crafter = UniversalPerturbation(
                    classifier,
                    attacker='fgsm',
                    delta=0.000001,
                    attacker_params={'targeted': False, 'eps': current_attack_eps},
                    max_iter=15,
                    eps=current_eps,
                    norm=np.inf
                )
                x_test_adv = adv_crafter.generate(x=x_subset)
            
            noise = adv_crafter.noise
            noise_tensor = torch.tensor(noise, dtype=torch.float32, device=device)
        
            file_name = f'Noise/Noise_{len(x_subset)}.npy'
            np.save(file_name, noise)
            
            if targeted_attack:
                # Evaluation with targeted noise
                val_loss_with_noise, val_acc_with_noise, success_rate, true_labels, pred_labels = evaluate_targeted_attack(
                    classifier, eval_loader, target_class, criterion, device, noise_tensor,
                    add_noise=True, remap=remap, eps_current=[current_attack_eps, current_eps], image_count=adv_image_count)
                print(f"With Noise {adv_image_count} - Val Loss: {val_loss_with_noise:.4f} - Val Acc: {val_acc_with_noise:.2f}% - Succ Rate: {success_rate:.2f}%")
            else:
                # Evaluation with noise
                val_loss_with_noise, val_acc_with_noise, true_labels, pred_labels = evaluate(
                    classifier, eval_loader, criterion, device, noise_tensor,
                    add_noise=True, remap=remap, eps_current=[current_attack_eps, current_eps], image_count=adv_image_count)
                print(f"With Noise {adv_image_count} - Val Loss: {val_loss_with_noise:.4f} - Val Acc: {val_acc_with_noise:.2f}%")
            
            resultFolder = 'ISIC2019/'
            plot_confusion_matrix(true_labels, pred_labels, f'confusion_matrix_{adv_image_count}_att{current_attack_eps}_eps{current_eps}')
            
            results_filename = fstr(config.get_experiment_path('evaluation_results_{adv_image_count}_att{current_attack_eps}_eps{current_eps}.txt'))
            save_results_to_file(results_filename, val_loss_no_noise, val_acc_no_noise, val_loss_with_noise, val_acc_with_noise, success_rate, True)
            
            classificationReportFileName = fstr(config.get_experiment_path('classification_report_{adv_image_count}_att{current_attack_eps}_eps{current_eps}.txt'))
            generate_classification_report(true_labels, pred_labels, classificationReportFileName)


end_time = time.time()

# Calculate the time taken
time_taken = end_time - start_time

# Print the time taken
print(f"Time taken: {time_taken} seconds")