# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 13:20:41 2023

@author: lkang
"""

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

def plot_confusion_matrix(true_labels,pred_labels,name):
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Display the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
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
                    make_adv_img(clean_image_for_psnr,noise_tensor,adv_img,f'ISIC2019/adv_img_att{eps_current[0]}eps{eps_current[1]}_{image_count}.jpg')
                    psnr(clean_image_np, adv_img_np,f'ISIC2019/psnr_att{eps_current[0]}_eps{eps_current[1]}.txt')
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


# transforms_list = []
# transforms_list.append(transforms.Resize((32,32)))
# transforms_list.append(transforms.ToTensor())
# transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
# transform = transforms.Compose(transforms_list)

# -------------------------------------------------------
# Parameters need to change

datapath = 'C:/Users/lkang/Documents/ISIC_2019_Training_Input/'
# testfile = 'C:/Users/lkang/Documents/New UAP/ISIC2019_test.csv'
trainfile = 'C:/Users/lkang/Documents/New UAP/ISIC2019_train_012.csv'
testfile = 'C:/Users/lkang/Documents/New UAP/ISIC2019_test_012.csv'
train_data_path = 'C:/Users/lkang/Documents/ISIC_2019_train/'
test_data_path = 'C:/Users/lkang/Documents/ISIC_2019_test/'
# testfile = 'C:/Users/lkang/Documents/Master_Code_backup/New UAP/ISIC2019_test_02.csv'
# testfile = 'C:/Users/lkang/Documents/Master_Code_backup/New UAP/ISIC2019_test_old.csv'
# adversarialFile = 'C:/Users/lkang/Documents/Master_Code_backup/New UAP/ISIC2019_train_02.csv'
# adversarialFile = 'C:/Users/lkang/Documents/Master_Code_backup/New UAP/ISIC2019_adversarial.csv'
# adversarialFile = 'C:/Users/lkang/Documents/New UAP/ISIC2019_train.csv'
adversarialFile = 'C:/Users/lkang/Documents/New UAP/ISIC2019_Adversarial_012.csv'
# adversarialFile = 'C:/Users/lkang/Documents/Master_Code_backup/New UAP/ISIC2019_Adversarial_02.csv'

# Load the previously trained model
# num_classes = 2
# num_classes = 4
num_classes = 3
# checkpoint = torch.load('C:/Users/lkang/Documents/Acc75_Epoch50_ BS16_3class/ISIC2019_morph.pth.tar',map_location ='cpu')
# checkpoint = torch.load('C:/Users/lkang/Documents/Acc73_Epoch100_BS16_4class/ISIC2019_morph.pth.tar',map_location ='cpu')
# checkpoint = torch.load('C:/Users/lkang/Documents/Cifar10_Acc76_BS16/cifar10_morph.pth.tar',map_location ='cpu')
# checkpoint = torch.load('C:/Users/lkang/Documents/ISIC_Model/Acc75_Epoch60_BS32_Clean/ISIC2019_morph.pth.tar',map_location ='cpu')

checkpoint = torch.load('C:/Users/lkang/Documents/ISIC_Model/Acc81_Epoch100_BS16_3class/ISIC2019_morph.pth.tar',map_location ='cpu')
# checkpoint = torch.load('C:/Users/lkang/Documents/04-16_skinV2_Resnet18_250_128_0.01/checkpoint.pth.tar',map_location ='cpu')
# Number of images to use for noise generation
image_count = 1773

# List of numbers of images to use for noise 
# image_counts = [1773,1596,1418,1241,1064,886,709,532,355,177,100,90,80,70,60]
# image_counts = [1773,1596,1418,1241,1064,886,709,532,355,177]
# image_counts = [100,90,80,70,60,50,40,30,20,10]
# image_counts = [10,9,8,7,6,5,4,3,2,1]
# image_counts = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
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

# test_transform = transforms.Compose([
#     transforms.Resize((32, 32)),  # Resize to match CIFAR-10 dimensions
#     transforms.RandomHorizontalFlip(),  # Data augmentation
#     transforms.ToTensor(),  # Convert images to PyTorch tensors
#     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])  # Normalize like CIFAR-10
# ])

# data_root = "/home/ubuntu/temps/"
# test_dataset = torchvision.datasets.CIFAR10(data_root,False, transform, download=True)
# train_dataset = torchvision.datasets.CIFAR10(data_root,True, transform, download=True)
# noise_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# eval_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# eval_dataset_length = len(test_dataset)
# print(f"Number of images in the evaluation datasetS: {eval_dataset_length}")

# train_dataset =  datasets.ImageFolder(train_data_path,test_transform)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# train_dataset_length = int(len(train_dataset))
# print(f"Number of images in the train dataset: {train_dataset_length}")

# eval_dataset = datasets.ImageFolder(test_data_path,test_transform)
# eval_dataset_length = len(eval_dataset)
# eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True)
# print(f"Number of images in the evaluation dataset: {eval_dataset_length}")

# adv_training_dataset = datasets.ImageFolder(train_data_path,test_transform)
# adv_training_loader = DataLoader(adv_training_dataset, batch_size=16, shuffle=True)
# adv_training_length = len(adv_training_dataset)
# print(f"Number of images in the adv training dataset: {adv_training_length}")

# Dataset for noise generation (including only the first image_count images)
# noise_dataset = ISICDataset(datapath, adversarialFile, 'test_data', transform=test_transform)
# train_dataset = ISICDataset(datapath, adversarialFile, 'train_data', transform=test_transform, one_hot_encode= True, num_classes=num_classes)
# # noise_dataset.data = noise_dataset.data[:image_count]
# # noise_dataset.labels = noise_dataset.labels[:image_count]
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# train_dataset_length = len(train_dataset)
# print(f"Number of images in the noise dataset: {train_dataset_length}")

# Dataset for evaluation (excluding the images used for noise generation)
eval_dataset = ISICDataset(datapath, testfile, 'test_data', transform=test_transform, one_hot_encode= False, num_classes=num_classes)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True)
eval_dataset_length = len(eval_dataset)
print(f"Number of images in the evaluation dataset: {eval_dataset_length}")

adv_training_dataset = ISICDataset(datapath, trainfile, 'train_data', transform=test_transform, one_hot_encode= True, num_classes=num_classes)
adv_training_loader = DataLoader(adv_training_dataset, batch_size=16, shuffle=True)
adv_training_length = len(adv_training_dataset)
print(f"Number of images in the adv training dataset: {adv_training_length}")

noise_dataset = ISICDataset(datapath, adversarialFile, 'train_data', transform=test_transform, one_hot_encode= False, num_classes=num_classes)
noise_loader = DataLoader(noise_dataset, batch_size=16, shuffle=True)
noise_length = len(noise_dataset)
print(f"Number of images in the adv training dataset: {noise_length}")

device = 'cuda'
model = resnet50()
# model = PreActResNet50()
# model = create_resnet(18, num_classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
# model.load_state_dict(checkpoint['netC'])
# model.load_state_dict(checkpoint['state_dict'])
model.train()
# model.eval()


# Define the loss function and the optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=5e-4)
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

# clean_classifier = PyTorchClassifier(
#     model=model,
#     loss=criterion,
#     input_shape=(3,256,256),
#     optimizer=optimizer,
#     nb_classes=num_classes,
#     device_type='gpu',
#     preprocessing=None
# )
# Universal Perturbation parameters
mean_inf_train = 0.57  # Modify as needed

# Load noise
# file_path = 'Noise/Noise_100.npy'
# noise = np.load(file_path)

# test(model,eval_loader,"clean")

# Evaluation without noise
# val_loss_no_noise, val_acc_no_noise, true_labels, pred_labels = evaluate(classifier, eval_loader, criterion, device,remap=remap)
# print(f"Without Noise - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%")
# plot_confusion_matrix(true_labels, pred_labels, "confusion_matrix_clean")
# save_results_to_file("clean_result.txt",val_loss_no_noise,val_acc_no_noise, 0, 0,0,targeted=False)
# classificationReportFileName = 'classification_report_clean.txt'
# generate_classification_report(true_labels,pred_labels,classificationReportFileName)


# Defense Testing
# FBF
# epsilon = 2.0 / 255.0
# trainer = AdversarialTrainerFBFPyTorch(classifier, eps=epsilon, use_amp=False)

# # Build a Keras image augmentation object and wrap it in ART
# art_datagen = PyTorchDataGenerator(iterator=adv_training_loader, size=adv_training_length, batch_size=16)

# trainer.fit_generator(art_datagen, nb_epochs=10)

#==============
# Adversarial trainer
# FastGradientMethod ProjectedGradientDescent
# test fgsm 0.4,0.004
# attack_fgm = ProjectedGradientDescent(
#     estimator=clean_classifier, 
#     eps=8.0/255.0
#     )

# pgd = ProjectedGradientDescent(classifier, eps=8 / 255, eps_step=2 / 255, max_iter=100, num_random_init=1)
# pgd = ProjectedGradientDescent(classifier, eps=0.04, eps_step=0.024, max_iter=100, num_random_init=1)
# pgd = ProjectedGradientDescent(estimator=clean_classifier, eps=8.0/255.0,batch_size=32)
# 
# trainer = AdversarialTrainer(
#     classifier=classifier, 
#     attacks=attack_fgm, 
#     ratio=1
#     )

# trainer = AdversarialTrainer(
#     classifier=classifier, 
#     # attacks=attack_fgm, 
#     attacks=pgd, 
#     ratio=1
#     )

adversarial_x, adversarial_y = adv_training_dataset[0:2500]
clean_x, clean_y = adv_training_dataset[2500:]
# train_x, train_y = train_dataset[0:]
# adversarial_x_adv = pgd.generate(adversarial_x)

adv_trainer = AdversarialTrainerMadryPGD(nb_epochs = 50,eps = 0.04,eps_step=1 / 255,classifier = classifier,batch_size=32)
adv_trainer.fit(adversarial_x,adversarial_y,nb_epochs=100)
# 
# adversarial_x_comb  = np.append(adversarial_x_adv,train_x,axis=0)
# adversarial_y       = np.append(adversarial_y,train_y,axis=0)

# if remap:
#     adversarial_y = remap_labels(adversarial_y)

# trainer.fit_generator(art_datagen,nb_epochs = 1)
# trainer.fit(
#     x=adversarial_x, 
#     y=adversarial_y,
#     nb_epochs=10,
#     batch_size=16
#     )

#========================

image_list = []

success_rate = 0

# classifier.fit(adversarial_x_adv, adversarial_y, batch_size= 16, nb_epochs= 50, verbose=True)

model.eval()

# Evaluation madry_pgd
madry_classifier = adv_trainer.get_classifier()
val_loss_no_noise, val_acc_no_noise, true_labels, pred_labels = evaluate(madry_classifier, eval_loader, criterion, device,remap=remap)
print(f"Without Noise after adv train - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%")
plot_confusion_matrix(true_labels, pred_labels, "confusion_matrix_clean_after_advtrain")
save_results_to_file("evaluation_result_after_adv_train.txt",val_loss_no_noise,val_acc_no_noise, 0, 0, 0,targeted=False)
classificationReportFileName = 'classification_report_clean_after_advtrain.txt'
generate_classification_report(true_labels,pred_labels,classificationReportFileName)

madry_classifier.fit(clean_x,clean_y, batch_size= 32, nb_epochs= 100, verbose=True)
val_loss_no_noise, val_acc_no_noise, true_labels, pred_labels = evaluate(madry_classifier, eval_loader, criterion, device,remap=remap)
print(f"Without Noise after adv train - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%")
plot_confusion_matrix(true_labels, pred_labels, "confusion_matrix_clean_after_advtrain_2")
save_results_to_file("evaluation_result_after_adv_train_2.txt",val_loss_no_noise,val_acc_no_noise, 0, 0, 0,targeted=False)
classificationReportFileName = 'classification_report_clean_after_advtrain_2.txt'
generate_classification_report(true_labels,pred_labels,classificationReportFileName)


# val_loss_no_noise, val_acc_no_noise, true_labels, pred_labels = evaluate(classifier, eval_loader, criterion, device,remap=remap)
# print(f"Without Noise - Val Loss: {val_loss_no_noise:.4f} - Val Acc: {val_acc_no_noise:.2f}%")
# plot_confusion_matrix(true_labels, pred_labels, "confusion_matrix_clean_after_advtrain")
# save_results_to_file("clean_reuslt.txt",val_loss_no_noise,val_acc_no_noise, 0, 0, 0,targeted=False)
# classificationReportFileName = 'classification_report_clean_after_advtrain.txt'
# generate_classification_report(true_labels,pred_labels,classificationReportFileName)
   
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
            
            # # # Loop through the DataLoader and collect individual images
            # for images, labels in noise_loader:
            #     for img in images:
            #         image_list.append(img)
            #         if len(image_list) >= image_count:
            #             break
            #     if len(image_list) >= image_count:
            #         break
                
            # for images, labels in noise_loader:
            #     for img, label in zip(images, labels):
            #         class_image_count[label.item()] += 1  # Increment the count for the current label
            #         image_list.append(img)
            
            #         # Check if the image_count is reached for any class
            #         if any(count >= image_count for count in class_image_count.values()):
            #             break
            
            #     # Break the outer loop if the image_count is reached for any class
            #     if any(count >= image_count for count in class_image_count.values()):
            #         break
            
            print(class_image_count)
            
            random.shuffle(image_list)
            # Concatenate exactly 'image_count' number of images
            x_subset = torch.stack(image_list[:image_count]).cpu().numpy()
            
            print(f"number of subset: {len(x_subset)}")
            if targeted_attack :
                #------------------------------------
                # Generate targeted attack
                adv_crafter = TargetedUniversalPerturbation(
                    classifier,
                    attacker='fgsm',
                    delta=0.000001,
                    attacker_params={'targeted': True, 'eps': current_attack_eps},
                    max_iter=15,
                    eps=current_eps,#mean_inf_train*0.02
                    norm=np.inf
                )
                # Specify the target class as an integer (e.g., 2 for "Basal cell carcinoma")
                target_class = 3
                
                # Create a one-hot encoded array of target labels
                target_labels = np.array([to_one_hot(target_class, num_classes) for _ in range(len(x_subset))])
            
                # Generate adversarial examples with the target class
                x_test_adv = adv_crafter.generate(x=x_subset, y=target_labels)
                #------------------------------
            else:
                #Generate universal attack
                adv_crafter = UniversalPerturbation(
                    classifier,
                    attacker='fgsm',
                    delta=0.000001,#0.000001,
                    attacker_params={'targeted': False, 'eps': current_attack_eps}, #eps = 0.001,0.0024
                    max_iter=15,
                    eps= current_eps,#0.04
                    norm=np.inf
                )
                x_test_adv = adv_crafter.generate(x=x_subset)
                
                # #-------------------------------------
            
            noise = adv_crafter.noise
            # noise = adv_crafter.noise[0, :]
            # noise = noise.astype(np.float32)
            noise_tensor = torch.tensor(noise, dtype=torch.float32, device=device)
        
            file_name = f'Noise/Noise_{len(x_subset)}.npy'
            np.save(file_name, noise)
            
            if targeted_attack :
                # Evaluation with targeted noise
                val_loss_with_noise, val_acc_with_noise, success_rate, true_labels, pred_labels = evaluate_targeted_attack(classifier, eval_loader, target_class, criterion, device, noise_tensor,
                                                                                                                            add_noise=True,
                                                                                                                            remap= remap,
                                                                                                                            eps_current=[current_attack_eps,current_eps],
                                                                                                                            image_count=image_count)
                print(f"With Noise {image_count} - Val Loss: {val_loss_with_noise:.4f} - Val Acc: {val_acc_with_noise:.2f}% - Succ Rate: {success_rate:.2f}%")
            else :
                # # Evaluation with noise
                val_loss_with_noise, val_acc_with_noise, true_labels, pred_labels = evaluate(classifier, eval_loader, criterion, device, noise_tensor, 
                                                                                              add_noise=True,
                                                                                              remap=remap,
                                                                                              eps_current=[current_attack_eps,current_eps],
                                                                                              image_count=image_count)
                print(f"With Noise {image_count} - Val Loss: {val_loss_with_noise:.4f} - Val Acc: {val_acc_with_noise:.2f}%")

            
            resultFolder = 'ISIC2019/'
            plot_confusion_matrix(true_labels, pred_labels, f'confusion_matrix_{image_count}_att{current_attack_eps}_eps{current_eps}')
            
            
            # results_filename = resultFolder+f'evaluation_results_{image_count}_{iteration}.txt'
            results_filename =f'evaluation_results_{image_count}_att{current_attack_eps}_eps{current_eps}.txt'
            save_results_to_file(results_filename, val_loss_no_noise, val_acc_no_noise, val_loss_with_noise, val_acc_with_noise,success_rate,True)
            
            classificationReportFileName = f'classification_report_{image_count}_att{current_attack_eps}_eps{current_eps}.txt'
            generate_classification_report(true_labels,pred_labels,classificationReportFileName)

