import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

import torch
import torch.nn as nn
import os
import time
import warnings
from torchvision.models import resnet50
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from utils.Utils import plot_confusion_matrix, generate_classification_report

warnings.filterwarnings("ignore")


def train(model, loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(loader):.4f}")


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            running_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    return running_loss / len(loader), 100 * correct / total, all_labels, all_preds


def main():
    start_time = time.time()

    train_data_path = str(config.get_data_path('chest_xray/train'))
    test_data_path = str(config.get_data_path('chest_xray/test'))
    num_classes = 2
    num_epochs = 100
    device = 'cuda'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_data_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    print(f"Train dataset size: {len(train_dataset)}")

    eval_dataset = datasets.ImageFolder(test_data_path, transform)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True)
    print(f"Eval dataset size: {len(eval_dataset)}")

    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

    train(model, train_loader, criterion, optimizer, device, num_epochs)

    eval_loss, eval_acc, all_labels, all_preds = evaluate(model, eval_loader, criterion, device)
    print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_acc:.2f}%")

    plot_confusion_matrix(all_labels, all_preds, 'CXRAY/confusion_matrix_train')
    generate_classification_report(
        all_labels, all_preds,
        str(config.get_experiment_path('CXRAY/classification_report_train.txt')),
    )

    save_path = str(config.get_model_path(f'chest_xray_epoch{num_epochs}_BS16.pth'))
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, save_path)
    print(f"Model saved to {save_path}")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == '__main__':
    main()
