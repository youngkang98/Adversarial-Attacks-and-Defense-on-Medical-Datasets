# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 19:10:13 2024

@author: lkang
"""
import torch
import torch.nn.utils.prune as prune
from torch.autograd import Variable
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch.nn.functional as F
import numpy as np

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
    
def test2(model, loader, name):
    model.cuda()
    model.eval()
    device = 'cuda'
    y_true = []
    y_pred = []
    y_score=[]
    correct = 0
    val_acc = 0
    with torch.no_grad():
        for _, data in enumerate(loader):
            images, labels = data
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            # pred = outputs.data.max(
            #     1, keepdim=True)[1]  # get the index of the max log-probability
            # correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
            probs = torch.softmax(outputs,dim=1)
            prediction = outputs.max(1, keepdim=True)[1]
            y_true.append(labels.cpu().numpy())
            y_pred.append(prediction.detach().cpu().numpy())
            y_score.append(probs.detach().cpu().numpy())

        y_true = np.concatenate(y_true)
        y_true_2D= F.one_hot(torch.from_numpy(y_true), num_classes=7).cpu().numpy()
        y_pred = np.concatenate(y_pred)
        y_score = np.concatenate(y_score)
        test_bacc = balanced_accuracy_score(y_true, y_pred)
        test_auc=roc_auc_score(y_true_2D, y_score)
            # probs = torch.softmax(outputs,dim=1)
            # prediction = outputs.max(1, keepdim=True)[1]
            # y_true.append(labels.cpu().numpy())
            # y_pred.append(prediction.detach().cpu().numpy())
            # y_score.append(probs.detach().cpu().numpy())
            # val_acc += torch.sum(y_pred == y_true)
    
    print(f'Balanced Accuracy:{test_bacc}')
    print(f'ROC AUC Accuracy:{test_bacc}')
    print('\n{}: Accuracy: {}/{} ({:.0f}%)\n'.format(name, correct,
          len(loader.dataset), 100. * correct / len(loader.dataset)))