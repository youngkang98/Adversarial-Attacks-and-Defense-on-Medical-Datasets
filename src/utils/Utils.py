import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for scripts without a display
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(true_labels, pred_labels, name):
    output_path = config.get_experiment_path(f'{name}.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()


def generate_classification_report(true_labels, pred_labels, filename):
    os.makedirs(os.path.dirname(str(filename)), exist_ok=True)
    report = classification_report(true_labels, pred_labels)
    print(report)
    with open(filename, 'w') as f:
        f.write(report)
