import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

import numpy as np
import torch

from utils.plot import make_adv_img
from utils.data import psnr

# Maps sparse ISIC label indices to contiguous class indices for 4-class experiments.
label_mapping = {0: 0, 1: 1, 2: 2, 4: 3}


def remap_labels(labels):
    return torch.tensor([label_mapping[label.item()] for label in labels])


def to_one_hot(label, num_classes):
    vec = np.zeros(num_classes)
    vec[label] = 1
    return vec


def evaluate(classifier, loader, criterion, device, noise_tensor=None,
             add_noise=False, remap=False, eps_current=(0, 0), image_count=0):
    classifier._model.eval()
    val_loss = 0.0
    val_acc = 0.0
    true_labels = []
    pred_labels = []
    first_image = True

    with torch.no_grad():
        for images, labels in loader:
            if remap:
                images, labels = images.to(device), remap_labels(labels).to(device)
            else:
                images, labels = images.to(device), labels.to(device)
            true_labels.extend(labels.cpu().numpy())

            if add_noise and noise_tensor is not None:
                clean_image_for_psnr = images[0].clone()
                images = images + noise_tensor
                adv_img = images[0]
                if first_image:
                    clean_np = np.transpose(clean_image_for_psnr.cpu().numpy(), (1, 2, 0))
                    adv_np = np.transpose(adv_img.cpu().numpy(), (1, 2, 0))
                    make_adv_img(
                        clean_image_for_psnr, noise_tensor, adv_img,
                        str(config.get_experiment_path(
                            f'adv_img_att{eps_current[0]}eps{eps_current[1]}_{image_count}.jpg'
                        ))
                    )
                    psnr(
                        clean_np, adv_np,
                        str(config.get_experiment_path(
                            f'psnr_att{eps_current[0]}_eps{eps_current[1]}.txt'
                        ))
                    )
                    first_image = False

            outputs = classifier.predict(images.cpu().numpy())
            outputs = torch.tensor(outputs).to(device)
            _, preds = torch.max(outputs, 1)
            pred_labels.extend(preds.cpu().numpy())
            val_acc += torch.sum(preds == labels.data)
            val_loss += criterion(outputs, labels).item() * images.size(0)

    val_loss /= len(loader.dataset)
    val_acc = val_acc.double() / len(loader.dataset)
    return val_loss, val_acc * 100, true_labels, pred_labels


def evaluate_targeted_attack(classifier, loader, target_class, criterion, device,
                             noise_tensor=None, add_noise=False, remap=False,
                             eps_current=(0, 0), image_count=0):
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
                images, labels = images.to(device), remap_labels(labels).to(device)
            else:
                images, labels = images.to(device), labels.to(device)
            true_labels.extend(labels.cpu().numpy())

            if add_noise and noise_tensor is not None:
                clean_image_for_psnr = images[0].clone()
                images = images + noise_tensor
                adv_img = images[0]
                if first_image:
                    clean_np = np.transpose(clean_image_for_psnr.cpu().numpy(), (1, 2, 0))
                    adv_np = np.transpose(adv_img.cpu().numpy(), (1, 2, 0))
                    make_adv_img(
                        clean_image_for_psnr, noise_tensor, adv_img,
                        str(config.get_experiment_path(
                            f'adv_img_att{eps_current[0]}eps{eps_current[1]}_{image_count}.jpg'
                        ))
                    )
                    psnr(
                        clean_np, adv_np,
                        str(config.get_experiment_path(
                            f'psnr_att{eps_current[0]}_eps{eps_current[1]}.txt'
                        ))
                    )
                    first_image = False

            outputs = classifier.predict(images.cpu().numpy())
            outputs = torch.tensor(outputs).to(device)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            pred_labels.extend(preds.cpu().numpy())
            val_acc += torch.sum(preds == labels.data)
            total_samples += len(labels)
            # Count only samples that CHANGED to target class (not ones already there).
            successful_attacks += torch.sum(
                (labels != target_class) & (preds == target_class)
            ).item()

    val_loss /= total_samples
    val_acc = (val_acc.double() / len(loader.dataset)) * 100
    success_rate = (successful_attacks / total_samples) * 100
    return val_loss, val_acc, success_rate, true_labels, pred_labels


def save_results_to_file(filename, val_loss_clean, val_acc_clean,
                         val_loss_adv, val_acc_adv, success_rate=0, targeted=False):
    os.makedirs(os.path.dirname(str(filename)), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(f"Without Noise - Val Loss: {val_loss_clean:.4f} - Val Acc: {val_acc_clean:.2f}%\n")
        if targeted:
            f.write(f"With Noise - Val Loss: {val_loss_adv:.4f} - Val Acc: {val_acc_adv:.2f}%"
                    f" - Succ Rate: {success_rate:.2f}%\n")
        else:
            f.write(f"With Noise - Val Loss: {val_loss_adv:.4f} - Val Acc: {val_acc_adv:.2f}%\n")
