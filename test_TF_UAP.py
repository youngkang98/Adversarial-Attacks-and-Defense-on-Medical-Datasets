import numpy as np
import os
import cv2
from keras.utils import np_utils
from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import TFV2Classifier
from dataloader import load_data
import tensorflow as tf

# Step 1: Load the ISIC dataset
datapath = 'C:/Users/lkang/Documents/ISIC_2019_Training_Input/'
trainfile = 'C:/Users/lkang/Documents/Master_Code_backup/Master_Code_backup/New UAP/ISIC2019_train.csv'
testfile = 'C:/Users/lkang/Documents/Master_Code_backup/Master_Code_backup/New UAP/ISIC2019_test.csv'
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_data(datapath, trainfile, testfile)

# Load the trained TensorFlow model
model_path = 'path_to_saved_tensorflow_model'  # You should replace this with the path to your saved TensorFlow model
model = tf.keras.models.load_model(model_path)

# Create the ART classifier for TensorFlow
classifier = TFV2Classifier(
    model=model,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    input_shape=(224, 224, 3),
    nb_classes=8
)

# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test[:1000])
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:1000], axis=1)) / len(y_test[:1000])
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
adv_crafter = UniversalPerturbation(
    classifier,
    attacker='fgsm',
    delta=0.000001,
    attacker_params={'targeted': False, 'eps': 0.001},
    max_iter=15,
    verbose=True)
x_test_adv = adv_crafter.generate(x=x_test[:1000])

# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:1000], axis=1)) / len(y_test[:1000])
print("Accuracy on adversarial train examples: {}%".format(accuracy * 100))

x_test_adv_noise = x_test + adv_crafter.noise;

# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv_noise)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test noise examples: {}%".format(accuracy * 100))
