# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 07:50:15 2023

@author: lkang
"""

import json
import os
import shutil
from time import time

import config
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet18
from networks.models import Denormalizer, NetC_MNIST, Normalizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
from utils.dataloader import PostTensorTransform, get_dataloader,CSVDataset
from utils.utils import progress_bar
from keras.preprocessing.image import ImageDataGenerator
from torchvision.transforms.functional import to_tensor

from torchvision import datasets, transforms, models


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
    if opt.dataset == "ISIC2019":
        netC = models.resnet50(pretrained=True)
        netC.fc = nn.Linear(netC.fc.in_features, opt.num_classes)
        netC = netC.to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def train(train_dl, noise_grid, identity_grid, tf_writer, opt):
    print(" Train:")
    rate_bd = opt.pc

    total_bd = 0
    numOfImage = 0
    
    train_dir = str(config.get_data_path('Echocardiogram/train'))
    img_size = (128, 128)
    batch_size = 32
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    dataLoaded = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', color_mode='grayscale')
    dataset = ImageFolderDataset(dataLoaded)
    train_folder = os.path.join(opt.temps,"train")
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    
    test_folder = os.path.join(opt.temps,"test")
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    
    try:
        for batch_idx, (inputs, targets) in enumerate(dataset):
    
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            # inputs = inputs.to(opt.device)
            
            bs = inputs.shape[0]
    
            # Create backdoor data
            num_bd = int(bs * 1)
            if total_bd > opt.maxBD:
                num_bd = 0
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
    
            inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
            
            i = 0
            
            for image in inputs_bd:
                pred = torch.argmax(targets[i].cpu()).item()
                imageName = "backdoor_train_" +str(pred)+"_"+ str(numOfImage) + ".png"
                path = os.path.join(train_folder,str(pred))
                if not os.path.exists(path):
                    os.makedirs(path)
                path = os.path.join(path, imageName)
                torchvision.utils.save_image(image, path, normalize=True)
                i+=1
                numOfImage += 1;
    except:
        print("errors")
    
    numOfImage = 0;
    
    test_dir = str(config.get_data_path('Echocardiogram/test'))
    img_size = (128, 128)
    batch_size = 32
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    dataLoaded = train_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', color_mode='grayscale')
    dataset = ImageFolderDataset(dataLoaded)
        
    try:
        for batch_idx, (inputs, targets) in enumerate(dataset):
    
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            # inputs = inputs.to(opt.device)
            
            bs = inputs.shape[0]
    
            # Create backdoor data
            num_bd = int(bs * 1)
            if total_bd > opt.maxBD:
                num_bd = 0
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
    
            inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
            
            i = 0
            
            for image in inputs_bd:
                pred = torch.argmax(targets[i].cpu()).item()
                imageName = "backdoor_test_" +str(pred)+"_"+ str(numOfImage) + ".png"
                path = os.path.join(test_folder,str(pred))
                if not os.path.exists(path):
                    os.makedirs(path)
                path = os.path.join(path, imageName)
                torchvision.utils.save_image(image, path, normalize=True)
                i+=1
                numOfImage += 1;
    except:
        print("errors")
      

class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, directory_iterator):
        self.directory_iterator = directory_iterator

    def __len__(self):
        return len(self.directory_iterator)

    def __getitem__(self, index):
        x, y = self.directory_iterator[index]
        x = torch.stack([to_tensor(x[i]) for i in range(len(x))])
        return x, torch.tensor(y)
def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    elif opt.dataset == 'ISIC2019':
        opt.num_classes = 8
    elif opt.dataset == 'Echo':
        opt.num_classes = 3
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    elif opt.dataset == "ISIC2019":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    elif opt.dataset == 'Echo':
        opt.input_height = 128
        opt.input_width = 128
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    train_dl = get_dataloader(opt, True)

    print("Train from scratch!!!")
    epoch_current = 0

    # Prepare grid
    ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.upsample(ins, size=opt.input_height, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
        .to(opt.device)
    )
    array1d = torch.linspace(-1, 1, steps=opt.input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)

    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
    os.makedirs(opt.log_dir)
    with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
        json.dump(opt.__dict__, f, indent=2)
    tf_writer = SummaryWriter(log_dir=opt.log_dir)

    # for epoch in range(epoch_current, opt.n_iters):
    #     print("Epoch {}:".format(epoch + 1))
    train(train_dl, noise_grid, identity_grid, tf_writer, opt)
        


if __name__ == "__main__":
    main()
