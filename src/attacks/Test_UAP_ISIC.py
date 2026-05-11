import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

import torch
import torch.nn as nn
import numpy as np
import random
import warnings
from collections import defaultdict
from math import floor
from torchvision.models import resnet50
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import UniversalPerturbation, TargetedUniversalPerturbation

from dataloader import ISICDataset
from utils.eval_utils import (
    evaluate, evaluate_targeted_attack, save_results_to_file, to_one_hot
)
from utils.Utils import plot_confusion_matrix, generate_classification_report

warnings.filterwarnings("ignore")


def main():
    # -------------------------------------------------------
    # Parameters
    datapath = str(config.get_data_path('ISIC2019'))
    testfile = str(config.get_data_path('ISIC2019_test.csv'))
    adversarialFile = str(config.get_data_path('ISIC2019_train.csv'))

    num_classes = 4
    checkpoint = torch.load(str(config.get_model_path('ISIC2019_morph.pth.tar')), map_location='cpu')

    image_counts = [1773, 1596, 1418, 1241, 1064, 886, 709, 532, 355, 177]
    eps = [0.04]
    attack_eps = [0.0024]
    remap = True
    targeted_attack = False
    target_class = 3

    # ----------------------------------------------------------
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    noise_dataset = ISICDataset(datapath, adversarialFile, 'test_data', transform=test_transform)
    noise_loader = DataLoader(noise_dataset, batch_size=16, shuffle=True)

    eval_dataset = ISICDataset(datapath, testfile, 'test_data', transform=test_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True)
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    device = 'cpu'
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['netC'])
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)
    optimizer.load_state_dict(checkpoint['optimizerC'])

    # input_shape reflects the actual tensor size after transforms (224x224)
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=num_classes,
        device_type='cpu',
        preprocessing=None,
    )

    image_list = []
    success_rate = 0

    # Baseline clean evaluation
    val_loss_clean, val_acc_clean, true_labels, pred_labels = evaluate(
        classifier, eval_loader, criterion, device, remap=remap
    )
    print(f"Clean - Val Loss: {val_loss_clean:.4f} - Val Acc: {val_acc_clean:.2f}%")
    plot_confusion_matrix(true_labels, pred_labels, 'confusion_matrix_clean')
    save_results_to_file(
        str(config.get_experiment_path('clean_result.txt')),
        val_loss_clean, val_acc_clean, 0, 0, targeted=False,
    )
    generate_classification_report(
        true_labels, pred_labels,
        str(config.get_experiment_path('classification_report_clean.txt')),
    )

    for current_attack_eps in attack_eps:
        for current_eps in eps:
            for image_count in image_counts:
                base_count = floor(image_count / num_classes)
                extra_images = image_count % num_classes
                class_image_count = defaultdict(int)
                image_list.clear()

                for images, labels in noise_loader:
                    for img, label in zip(images, labels):
                        label_idx = label.item()
                        allowed = base_count + (1 if label_idx < extra_images else 0)
                        if class_image_count[label_idx] < allowed:
                            image_list.append(img)
                            class_image_count[label_idx] += 1
                        if sum(class_image_count.values()) >= image_count:
                            break
                    if sum(class_image_count.values()) >= image_count:
                        break

                print(f"Class counts: {dict(class_image_count)}")
                random.shuffle(image_list)
                x_subset = torch.stack(image_list[:image_count]).cpu().numpy()
                print(f"Subset size: {len(x_subset)}")

                if targeted_attack:
                    adv_crafter = TargetedUniversalPerturbation(
                        classifier,
                        attacker='fgsm',
                        delta=0.000001,
                        attacker_params={'targeted': True, 'eps': current_attack_eps},
                        max_iter=15,
                        eps=current_eps,
                        norm=np.inf,
                    )
                    target_labels = np.array([to_one_hot(target_class, num_classes)
                                              for _ in range(len(x_subset))])
                    adv_crafter.generate(x=x_subset, y=target_labels)
                else:
                    adv_crafter = UniversalPerturbation(
                        classifier,
                        attacker='fgsm',
                        delta=0.000001,
                        attacker_params={'targeted': False, 'eps': current_attack_eps},
                        max_iter=15,
                        eps=current_eps,
                        norm=np.inf,
                    )
                    adv_crafter.generate(x=x_subset)

                noise = adv_crafter.noise
                noise_tensor = torch.tensor(noise, dtype=torch.float32, device=device)
                np.save(str(config.get_experiment_path(f'Noise_{image_count}.npy')), noise)

                eps_tag = f'att{current_attack_eps}_eps{current_eps}'

                if targeted_attack:
                    val_loss_adv, val_acc_adv, success_rate, true_labels, pred_labels = (
                        evaluate_targeted_attack(
                            classifier, eval_loader, target_class, criterion, device,
                            noise_tensor, add_noise=True, remap=remap,
                            eps_current=[current_attack_eps, current_eps],
                            image_count=image_count,
                        )
                    )
                    print(f"With Noise {image_count} - Loss: {val_loss_adv:.4f}"
                          f" - Acc: {val_acc_adv:.2f}% - Succ: {success_rate:.2f}%")
                else:
                    val_loss_adv, val_acc_adv, true_labels, pred_labels = evaluate(
                        classifier, eval_loader, criterion, device, noise_tensor,
                        add_noise=True, remap=remap,
                        eps_current=[current_attack_eps, current_eps],
                        image_count=image_count,
                    )
                    print(f"With Noise {image_count} - Loss: {val_loss_adv:.4f}"
                          f" - Acc: {val_acc_adv:.2f}%")

                plot_confusion_matrix(
                    true_labels, pred_labels,
                    f'confusion_matrix_{image_count}_{eps_tag}',
                )
                save_results_to_file(
                    str(config.get_experiment_path(
                        f'evaluation_results_{image_count}_{eps_tag}.txt'
                    )),
                    val_loss_clean, val_acc_clean, val_loss_adv, val_acc_adv,
                    success_rate, targeted=targeted_attack,
                )
                generate_classification_report(
                    true_labels, pred_labels,
                    str(config.get_experiment_path(
                        f'classification_report_{image_count}_{eps_tag}.txt'
                    )),
                )


if __name__ == '__main__':
    main()
