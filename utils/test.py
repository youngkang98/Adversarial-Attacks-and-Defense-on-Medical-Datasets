# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 19:10:13 2024

@author: lkang
"""
import torch
import torch.nn.utils.prune as prune
from torch.autograd import Variable
from cbam.imagenet import create_resnet
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

def test(model, loader, name):
    model.cuda()
    model.eval()

    correct = 0
    for data, target in loader:

        data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\n{}: Accuracy: {}/{} ({:.0f}%)\n'.format(name, correct,
          len(loader.dataset), 100. * correct / len(loader.dataset)))