# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 13:20:41 2023

@author: lkang
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from dataloader import ISICDataset, OCTDataset
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import UniversalPerturbation,TargetedUniversalPerturbation
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
from math import floor
import os
from utils.plot import make_adv_img
from utils.data import psnr

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
# label_mapping = {0: 0, 1: 1, 2: 2, 4: 3}
label_mapping = {0: 0, 2: 1}

def remap_labels(labels):
    # Use the label_mapping dictionary to remap the labels
    return torch.tensor([label_mapping[label.item()] for label in labels])

# Update your evaluate function
def evaluate(classifier, loader, criterion, device, noise_tensor=None, add_noise=False, remap = False,eps_current=[0,0]):
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
                    make_adv_img(clean_image_for_psnr,noise_tensor,adv_img,f'{current_dataset}/adv_img_att{eps_current[0]}eps{eps_current[1]}.jpg')
                    psnr(clean_image_np, adv_img_np,f'{current_dataset}/psnr_att{eps_current[0]}_eps{eps_current[1]}.txt')
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

def evaluate_targeted_attack(classifier, loader, target_class, criterion, device, noise_tensor=None, add_noise=False):
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
            # images, labels = images.to(device), remap_labels(labels).to(device)  # Remap labels here
            images, labels = images.to(device), labels.to(device)
            true_labels.extend(labels.cpu().numpy())
            
            # Add noise if add_noise is True
            if add_noise and noise_tensor is not None:
                images += noise_tensor

            outputs = classifier.predict(images.cpu().numpy())
            outputs = torch.tensor(outputs).to(device)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            pred_labels.extend(preds.cpu().numpy())
            val_acc += torch.sum(preds == labels.data)

            total_samples += len(labels)
            # successful_attacks += torch.sum((preds == target_class).int()).item()
            successful_attacks += torch.sum(preds == target_class).item()

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
current_dataset = "OCT"
datapath = str(config.get_data_path('OCT'))

num_classes = 4

# checkpoint = torch.load(str(config.get_model_path('OCT_morph.pth.tar')),map_location ='cpu')
checkpoint = torch.load(str(config.get_model_path('OCT_morph.pth.tar')),map_location ='cpu')

# List of numbers of images to use for noise 
# image_counts = [1773,1596,1418,1241,1064,886,709,532,355,177,100,90,80,70,60]
# image_counts = [1773,1596,1418,1241,1064,886,709,532,355,177]
# image_counts = [100,90,80,70,60,50,40,30,20,10]
# image_counts = [10,9,8,7,6,5,4,3,2,1]
# image_counts = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
image_counts = [1000]

# iterations = [35,30,25,20,15,10]
iterations = [25]
eps = [0.0005,0.001,0.002,0.005,0.01,0.02,0.03,0.04,0.05]
attack_eps = [0.0005,0.001,0.002,0.005,0.01,0.02,0.03,0.04,0.05]
remap = False

# ----------------------------------------------------------

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

noise_dataset = OCTDataset(datapath= os.path.join(datapath, 'train'),transform=test_transform)
noise_loader = DataLoader(noise_dataset)

# Dataset for evaluation (excluding the images used for noise generation)
eval_dataset = OCTDataset(datapath= os.path.join(datapath, 'test'),transform=test_transform)
eval_loader = DataLoader(eval_dataset, batch_size= 8)
eval_dataset_length = len(eval_dataset)
print(f"Number of images in the evaluation dataset {eval_dataset_length}")


device = 'cpu'
model = resnet50()
# model = torch.load(str(config.get_model_path('OCT_morph.pth.tar')),map_location ='cpu')
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
model.load_state_dict(checkpoint['net'])
model.eval()

# Define the loss function and the optimizer
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer.load_state_dict(checkpoint['optimizerC'])

# Create the ART classifier
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 224, 224),
    nb_classes=num_classes,
    device_type='gpu'
)
# Universal Perturbation parameters
mean_inf_train = 0.57  # Modify as needed


# Load noise
# file_path = 'Noise/Noise_100.npy'
# noise = np.load(file_path)

image_list = []

success_rate = 0

# Evaluation without noise
val_loss_no_noise, val_acc_no_noise, true_labels, pred_labels = evaluate(classifier, eval_loader, criterion, device)
print(f"Without Noise - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%")
plot_confusion_matrix(true_labels, pred_labels, "confusion_matrix_clean")
save_results_to_file(str(config.get_experiment_path('clean_reuslt.txt')),val_loss_no_noise,val_acc_no_noise, 0, 0,0,targeted=False)
classificationReportFileName = str(config.get_experiment_path('classification_report_clean.txt'))
generate_classification_report(true_labels,pred_labels,classificationReportFileName)
   
# Loop through the different numbers of images
for current_attack_eps in attack_eps:
    for current_eps in eps:
        for image_count in image_counts:
            # Calculate base count per class for current target
            base_count = floor(image_count / num_classes)
            # Calculate how many classes need an extra image to reach the target
            extra_images = image_count % num_classes
    
            # Initialize counts and image list for the current subset
            class_image_count = defaultdict(int)
            image_list.clear()
            
            for images, labels in noise_loader:
                for img, label in zip(images, labels):
                    label_index = label.item()
                    # Calculate the allowed count for the current class
                    allowed_count = base_count + (1 if label_index < extra_images else 0)
        
                    # Only add the image if the class count hasn't reached the limit
                    if class_image_count[label_index] < allowed_count:
                        image_list.append(img)
                        class_image_count[label_index] += 1
        
                    # Check if we have reached the target count
                    if sum(class_image_count.values()) >= image_count:
                        break
                if sum(class_image_count.values()) >= image_count:
                    break
            
            print(class_image_count)
            
            random.shuffle(image_list)
            # Concatenate exactly 'image_count' number of images
            x_subset = torch.stack(image_list[:image_count]).cpu().numpy()
            
            print(f"number of subset: {len(x_subset)}")
            # Generate targeted attack
            # adv_crafter = TargetedUniversalPerturbation(
            #     classifier,
            #     attacker='fgsm',
            #     delta=0.000001,
            #     attacker_params={'targeted': True, 'eps': 0.001},
            #     max_iter=iteration,
            #     eps=0.001,#mean_inf_train*0.02
            #     norm=np.inf
            # )
            # # Specify the target class as an integer (e.g., 2 for "Basal cell carcinoma")
            # target_class = 1
            
            # # Create a one-hot encoded array of target labels
            # target_labels = np.array([to_one_hot(target_class, num_classes) for _ in range(len(x_subset))])
        
            # # Generate adversarial examples with the target class
            # x_test_adv = adv_crafter.generate(x=x_subset, y=target_labels)
            # #------------------------------
            
            
            #Generate universal attack
            adv_crafter = UniversalPerturbation(
                classifier,
                attacker='fgsm',
                delta=0.00001,#0.000001,
                attacker_params={'targeted': False, 'eps': current_attack_eps},#0.0024
                max_iter=25,
                eps= current_eps,#0.04,#mean_inf_train * 0.02,
                norm=np.inf
            )
            x_test_adv = adv_crafter.generate(x=x_subset)
            
            noise = adv_crafter.noise
            # noise = adv_crafter.noise[0, :]
            # noise = noise.astype(np.float32)
            noise_tensor = torch.tensor(noise, dtype=torch.float32, device=device)
        
            file_name = f'Noise/Noise_{len(x_subset)}.npy'
            np.save(file_name, noise)
            
            # Evaluation with noise
            val_loss_with_noise, val_acc_with_noise, true_labels, pred_labels = evaluate(classifier, eval_loader, criterion, device, noise_tensor, add_noise=True,remap=remap,eps_current=[current_attack_eps,current_eps])
            print(f"With Noise {image_count} - Val Loss: {val_loss_with_noise:.4f} - Val Acc: {val_acc_with_noise:.2f}%")
            # Evaluation with targeted noise
            # val_loss_with_noise, val_acc_with_noise, success_rate, true_labels, pred_labels = evaluate_targeted_attack(classifier, eval_loader, target_class, criterion, device, noise_tensor, add_noise=True)
            # print(f"With Noise {image_count} - Val Loss: {val_loss_with_noise:.4f} - Val Acc: {val_acc_with_noise:.2f}% - Succ Rate: {success_rate:.2f}%")
            
            resultFolder = f'{current_dataset}/'
            plot_confusion_matrix(true_labels, pred_labels, resultFolder+f'confusion_matrix_{image_count}_att{current_attack_eps}_eps{current_eps}')
            
            
            # results_filename = resultFolder+fstr(config.get_experiment_path('evaluation_results_{image_count}_{iteration}.txt'))
            results_filename = resultFolder+fstr(config.get_experiment_path('evaluation_results_{image_count}_att{current_attack_eps}_eps{current_eps}.txt'))
            save_results_to_file(results_filename, val_loss_no_noise, val_acc_no_noise, val_loss_with_noise, val_acc_with_noise,success_rate,True)
            
            classificationReportFileName = resultFolder+fstr(config.get_experiment_path('classification_report_{image_count}_att{current_attack_eps}_eps{current_eps}.txt'))
            generate_classification_report(true_labels,pred_labels,classificationReportFileName)


