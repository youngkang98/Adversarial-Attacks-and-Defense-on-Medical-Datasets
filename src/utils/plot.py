# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:19:18 2023

@author: lkang
"""
import itertools

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
from PIL import Image



def make_adv_img_ori(clean_img, noise, adv_img, save_file_name):
    # clean
    im_clean = (clean_img * 128.0) + 128.0
    im_clean = np.squeeze(np.clip(im_clean, 0, 255).astype(np.uint8))
    # noise
    im_noise = (noise - noise.min()) / \
        (noise.max() - noise.min()) * 128.0
    im_noise = np.squeeze(im_noise.astype(np.uint8))
    # adv
    im_adv = (adv_img * 128.0) + 128.0
    im_adv = np.squeeze(np.clip(im_adv, 0, 255).astype(np.uint8))
    # all
    img_all = np.concatenate((im_clean, im_noise, im_adv), axis=1)
    img_all = Image.fromarray(np.uint8(img_all))
    img_all.save(save_file_name)
    

def make_adv_img(clean_img, noise, adv_img, save_file_name):
    # Function to convert tensor to numpy array if it's a tensor
    def tensor_to_numpy(tensor):
        return tensor.cpu().detach().numpy()

    # Check if input is tensor and convert to numpy array
    if torch.is_tensor(clean_img):
        clean_img = tensor_to_numpy(clean_img)
    if torch.is_tensor(noise):
        # noise = tensor_to_numpy(noise)
        noise = noise[0].cpu().detach().numpy()
    if torch.is_tensor(adv_img):
        adv_img = tensor_to_numpy(adv_img)

    # Normalize and process the images
    # Assuming the input images are in the range [0, 1]
    # clean
    im_clean = (clean_img * 255).astype(np.uint8)
    im_clean = np.transpose(im_clean, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    # noise
    im_noise = (noise * 255).astype(np.uint8)
    im_noise = np.transpose(im_noise, (1, 2, 0))
    # adv
    im_adv = (adv_img * 255).astype(np.uint8)
    im_adv = np.transpose(im_adv, (1, 2, 0))

    # Concatenate and save the image
    img_all = np.concatenate((im_clean, im_noise, im_adv), axis=1)
    img_all = Image.fromarray(img_all)
    img_all.save(save_file_name)
