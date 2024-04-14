# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:07:20 2023

@author: lkang
"""

import numpy as np
import torch

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def psnr(imageA, imageB,filename):
    max_pixel = 255.0
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        result = float('inf')
    result = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    with open(filename, 'w') as file:
        file.write(f"psnr:{result}")