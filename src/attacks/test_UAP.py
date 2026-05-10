# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:10:15 2023

@author: lkang
"""

"""
The script demonstrates a simple example of using ART with TensorFlow v1.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

import numpy as np

from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
from dataloader import load_data, load_data_without_train
from torchvision import models
import torch
import torch.nn as nn
import os
import gc
torch.cuda.empty_cache()
gc.collect()
# import torch.optim as optim

# Step 1: Load the MNIST dataset
datapath= str(config.get_data_path('ISIC2019'))
trainfile = str(config.get_data_path('ISIC2019_train.csv'))
testfile = str(config.get_data_path('ISIC2019_test.csv'))
# (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value, mean_l2_train, mean_inf_train = load_data(datapath, trainfile, testfile)
(x_test, y_test), min_pixel_value, max_pixel_value, mean_l2_train, mean_inf_train = load_data_without_train(datapath, testfile)
print(mean_inf_train)
# (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_iris()
# x_train = get_isic()

# from tensorflow.keras.applications.densenet import DenseNet201
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

num_classes = 8
checkpoint = torch.load(str(config.get_model_path('ISIC2019_morph.pth.tar')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
#model = model.to(device)
model = model.to(torch.device('cpu'))

model.load_state_dict(checkpoint['netC'])

# Define the loss function and the optimizer
criterion = torch.nn.CrossEntropyLoss()  # Replace this with the loss function you used in training
optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Replace this with the optimizer you used in training
optimizer.load_state_dict(checkpoint['optimizerC'])

# Create the ART classifier
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 224, 224),  # Replace this with the shape of your inputs
    nb_classes=8,  # Replace this with the number of classes in your problem device_type='cpu'
    device_type='cpu'
)

del criterion
del optimizer
del checkpoint
del model

# Step 4: Train the ART classifier
# classifier.fit(x_train, y_train, batch_size=64, epochs=200)
#classifier.fit(x_train, y_train, batch_size=64, nb_epochs=200)



# # Step 5: Evaluate the ART classifier on benign test examples
# print(" Step 5: Evaluate the ART classifier on benign test examples")
# predictions = classifier.predict(x_test[:1])
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:1], axis=1)) / len(y_test[:1])
# print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# # Step 6: Generate adversarial test examples

# #attack = UniversalPerturbation(classifier,attacker='fgsm',delta=0.001,eps=0.001,max_iter=15)
# attack = UniversalPerturbation(classifier,attacker='fgsm',eps=0.2)

adv_crafter = UniversalPerturbation(
    classifier,
    attacker='fgsm',
    delta=0.000001,
    attacker_params={'targeted': False, 'eps': 0.001},
    max_iter=15,
    eps= mean_inf_train * 0.02,
    norm=np.inf)


print("Step 6: Generate adversarial test examples")
x_test_adv = adv_crafter.generate(x=x_test[:1700])

np.save('Noise/Noise_1700.npy', adv_crafter.noise)
# adv_crafter = UniversalPerturbation(
#     classifier,
#     attacker='fgsm',
#     delta=0.000001,
#     attacker_params={'targeted': False, 'eps': 0.001},
#     max_iter=15,
#     verbose=True)

# If the directory does not exist, create it
# if not os.path.exists('Noise'):
#     os.makedirs('Noise')

# # Save the array
# # np.save('Noise/Noise_100.npy', adv_crafter.noise)
# file_path = 'Noise/Noise_100.npy'
# noise = np.load(file_path)

# # # Step 7: Evaluate the ART classifier on adversarial test examples
# # print("Step 7: Evaluate the ART classifier on adversarial test examples")
# # predictions = classifier.predict(x_test_adv[:1000])
# # accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:1000], axis=1)) / len(y_test[:1000])
# # print("Accuracy on adversarial train examples: {}%".format(accuracy * 100))
# print("Step 7: Evaluate the ART classifier on adversarial test examples")
# x_test_adv_noise = x_test[:5000] + noise;

# # Step 7: Evaluate the ART classifier on adversarial test examples

# predictions = classifier.predict(x_test_adv_noise)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:5000], axis=1)) / len(y_test[:5000])
# print("Accuracy on adversarial test noise examples: {}%".format(accuracy * 100))