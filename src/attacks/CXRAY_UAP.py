import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

import torch
import torch.nn as nn
import numpy as np
import random
import time
import warnings
from collections import defaultdict
from torchvision.models import resnet50
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import UniversalPerturbation, TargetedUniversalPerturbation

from dataloader import DatasetSeprateByClass
from utils.eval_utils import (
    evaluate, evaluate_targeted_attack, save_results_to_file, to_one_hot
)
from utils.Utils import plot_confusion_matrix, generate_classification_report

warnings.filterwarnings("ignore")


def main():
    start_time = time.time()

    # -------------------------------------------------------
    # Parameters
    train_data_path = str(config.get_data_path('chest_xray/train'))
    test_data_path = str(config.get_data_path('chest_xray/test'))
    testfile = str(config.get_data_path('CXRAY-test.csv'))
    adversarialFile = str(config.get_data_path('CXRAY-train.csv'))

    num_classes = 2
    image_counts = [525, 473, 420, 368, 315, 263, 210, 158, 105, 53]
    checkpoint = torch.load(
        str(config.get_model_path('chest_xray_epoch100_BS16.pth')), map_location='cpu'
    )

    eps = [0.04]
    attack_eps = [0.0024]
    adv_percentages = [0.1]
    remap = False
    targeted_attack = False
    target_class = 1

    # ----------------------------------------------------------
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    eval_dataset = DatasetSeprateByClass(
        test_data_path, testfile, 'test_data',
        transform=test_transform, one_hot_encode=False, num_classes=num_classes,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True)
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    noise_dataset = DatasetSeprateByClass(
        train_data_path, adversarialFile, 'train_data',
        transform=test_transform, one_hot_encode=False, num_classes=num_classes,
    )
    noise_loader = DataLoader(noise_dataset, batch_size=8, shuffle=True)
    noise_length = len(noise_dataset)
    print(f"Noise/training dataset size: {noise_length}")

    device = 'cuda'
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-3, momentum=0.9, weight_decay=5e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # input_shape reflects tensor size after transforms (CenterCrop 224)
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=(3, 224, 224),
        optimizer=optimizer,
        nb_classes=num_classes,
        device_type='gpu',
        preprocessing=None,
    )

    success_rate = 0

    # Baseline clean evaluation
    val_loss_clean, val_acc_clean, true_labels, pred_labels = evaluate(
        classifier, eval_loader, criterion, device, remap=remap
    )
    print(f"Clean - Val Loss: {val_loss_clean:.4f} - Val Acc: {val_acc_clean:.2f}%")
    plot_confusion_matrix(true_labels, pred_labels, 'CXRAY/confusion_matrix_clean')
    save_results_to_file(
        str(config.get_experiment_path('CXRAY/evaluation_result_clean.txt')),
        val_loss_clean, val_acc_clean, 0, 0, targeted=False,
    )
    generate_classification_report(
        true_labels, pred_labels,
        str(config.get_experiment_path('CXRAY/classification_report_clean.txt')),
    )

    # Collect images by class for percentage-based subset selection
    class_images = defaultdict(list)
    for images, labels in noise_loader:
        for img, label in zip(images, labels):
            class_images[label.item()].append(img)

    for current_attack_eps in attack_eps:
        for current_eps in eps:
            for percentage in adv_percentages:
                adv_image_count = int(noise_length * percentage)

                adv_images, adv_labels = [], []
                for class_idx, imgs in class_images.items():
                    split_point = int(len(imgs) * percentage)
                    random.shuffle(imgs)
                    adv_images.extend(imgs[:split_point])
                    adv_labels.extend([torch.tensor(class_idx)] * split_point)

                print(f"Adv subset distribution: "
                      f"{defaultdict(int, {i: adv_labels.count(torch.tensor(i)) for i in range(num_classes)})}")

                random.shuffle(adv_images)
                x_subset = torch.stack(adv_images[:adv_image_count]).cpu().numpy()
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
                np.save(str(config.get_experiment_path(f'Noise_{adv_image_count}.npy')), noise)

                eps_tag = f'att{current_attack_eps}_eps{current_eps}'

                if targeted_attack:
                    val_loss_adv, val_acc_adv, success_rate, true_labels, pred_labels = (
                        evaluate_targeted_attack(
                            classifier, eval_loader, target_class, criterion, device,
                            noise_tensor, add_noise=True, remap=remap,
                            eps_current=[current_attack_eps, current_eps],
                            image_count=adv_image_count,
                        )
                    )
                    print(f"With Noise {adv_image_count} - Loss: {val_loss_adv:.4f}"
                          f" - Acc: {val_acc_adv:.2f}% - Succ: {success_rate:.2f}%")
                else:
                    val_loss_adv, val_acc_adv, true_labels, pred_labels = evaluate(
                        classifier, eval_loader, criterion, device, noise_tensor,
                        add_noise=True, remap=remap,
                        eps_current=[current_attack_eps, current_eps],
                        image_count=adv_image_count,
                    )
                    print(f"With Noise {adv_image_count} - Loss: {val_loss_adv:.4f}"
                          f" - Acc: {val_acc_adv:.2f}%")

                plot_confusion_matrix(
                    true_labels, pred_labels,
                    f'CXRAY/confusion_matrix_{adv_image_count}_{eps_tag}',
                )
                save_results_to_file(
                    str(config.get_experiment_path(
                        f'CXRAY/evaluation_results_{adv_image_count}_{eps_tag}.txt'
                    )),
                    val_loss_clean, val_acc_clean, val_loss_adv, val_acc_adv,
                    success_rate, targeted=targeted_attack,
                )
                generate_classification_report(
                    true_labels, pred_labels,
                    str(config.get_experiment_path(
                        f'CXRAY/classification_report_{adv_image_count}_{eps_tag}.txt'
                    )),
                )

    print(f"Time taken: {time.time() - start_time:.1f}s")


if __name__ == '__main__':
    main()
