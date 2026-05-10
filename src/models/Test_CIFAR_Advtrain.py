# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 08:05:39 2024

@author: lkang
"""

"""
This is an example of how to use ART and Keras to perform adversarial training using data generators for CIFAR10
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

from torchvision.models import resnet50
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import keras
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2


from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier, PyTorchClassifier
from art.data_generators import KerasDataGenerator
from art.defences.trainer import AdversarialTrainer
from art.utils import load_cifar10


# Example LeNet classifier architecture with Keras & ART
# To obtain good performance in adversarial training on CIFAR-10, use a larger architecture
def build_model(input_shape=(32, 32, 3), nb_classes=10):
    img_input = Input(shape=input_shape)
    conv2d_1 = Conv2D(
        6,
        (5, 5),
        padding="valid",
        kernel_regularizer=l2(0.0001),
        activation="relu",
        kernel_initializer="he_normal",
        input_shape=input_shape,
    )(img_input)
    conv2d_1_bn = BatchNormalization()(conv2d_1)
    conv2d_1_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv2d_1_bn)
    conv2d_2 = Conv2D(16, (5, 5), padding="valid", activation="relu", kernel_initializer="he_normal")(conv2d_1_pool)
    conv2d_2_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv2d_2)
    flatten_1 = Flatten()(conv2d_2_pool)
    dense_1 = Dense(120, activation="relu", kernel_initializer="he_normal")(flatten_1)
    dense_2 = Dense(84, activation="relu", kernel_initializer="he_normal")(dense_1)
    img_output = Dense(nb_classes, activation="softmax", kernel_initializer="he_normal")(dense_2)
    model = Model(img_input, img_output)

    model.compile(
        loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=["accuracy"]
    )

    return model


# Load data and normalize
(x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()

# Build a Keras image augmentation object and wrap it in ART
batch_size = 50
datagen = ImageDataGenerator(
    horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125, fill_mode="constant", cval=0.0
)
datagen.fit(x_train)
art_datagen = KerasDataGenerator(
    datagen.flow(x=x_train, y=y_train, batch_size=batch_size, shuffle=True),
    size=x_train.shape[0],
    batch_size=batch_size,
)

loss=keras.losses.categorical_crossentropy 
optimizer=keras.optimizers.Adam(lr=0.001)

# Create a toy Keras CNN architecture & wrap it under ART interface
# model = resnet50()
# classifier = PyTorchClassifier(
#     model=model,
#     nb_classes=10,
#     device_type='gpu',
#     loss = loss,
#     optimizer=optimizer,
#     input_shape=(3,32, 32),
#     preprocessing=None,
#     clip_values=(0,1),
# )
classifier = KerasClassifier(build_model(), clip_values=(0, 1), use_logits=False)

# Create attack for adversarial trainer; here, we use 2 attacks, both crafting adv examples on the target model
pgd = ProjectedGradientDescent(classifier, eps=8 / 255, eps_step=2 / 255, max_iter=10, num_random_init=1)

# Create some adversarial samples for evaluation
x_test_pgd = pgd.generate(x_test)

classifier.fit(x_train,y_train,batch_size,nb_epochs=100,verbose=True)

# Evaluate the clean model on clean test set
labels_true = np.argmax(y_test, axis=1)
labels_test = np.argmax(classifier.predict(x_test), axis=1)
print("Accuracy clean test set: %.2f%%" % (np.sum(labels_test == labels_true) / x_test.shape[0] * 100))

# Evaluate the clean model on original adversarial samples
labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
print(
    "Accuracy on original PGD adversarial samples before adv train: %.2f%%" % (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100)
)

# Create adversarial trainer and perform adversarial training
adv_trainer = AdversarialTrainer(classifier, attacks=pgd, ratio=1.0)
adv_trainer.fit_generator(art_datagen, nb_epochs=83)

# Evaluate the adversarially trained model on clean test set
labels_true = np.argmax(y_test, axis=1)
labels_test = np.argmax(classifier.predict(x_test), axis=1)
print("Accuracy after adv test set: %.2f%%" % (np.sum(labels_test == labels_true) / x_test.shape[0] * 100))

# Evaluate the adversarially trained model on original adversarial samples
labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
print(
    "Accuracy on original PGD adversarial samples: %.2f%%" % (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100)
)

# Evaluate the adversarially trained model on fresh adversarial samples produced on the adversarially trained model
x_test_pgd = pgd.generate(x_test)
labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
print("Accuracy on new PGD adversarial samples: %.2f%%" % (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100))