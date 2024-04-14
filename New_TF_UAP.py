import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("Using GPU")
    except RuntimeError as e:
        print(e)



import os
import numpy as np
from PIL import Image
import csv
from dataloader import ISICDatasetTF, load_data

# Model, optimizer, and loss definition
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
criterion = tf.keras.losses.CategoricalCrossentropy()
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(8, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=x)


# Define the training and evaluation functions
def train(model, train_ds, criterion, optimizer):
    train_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for images, labels in train_ds:
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            loss = criterion(labels, outputs)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss += loss.numpy() * images.shape[0]
        correct_predictions += tf.math.reduce_sum(tf.cast(tf.math.argmax(outputs, axis=1) == tf.math.argmax(labels, axis=1), tf.float32))
        total_predictions += images.shape[0]
    train_loss /= total_predictions
    train_acc = correct_predictions / total_predictions
    return train_loss, train_acc

def evaluate(model, val_ds, criterion):
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for images, labels in val_ds:
        outputs = model(images, training=False)
        loss = criterion(labels, outputs)
        val_loss += loss.numpy() * images.shape[0]
        correct_predictions += tf.math.reduce_sum(tf.math.argmax(outputs, axis=1) == tf.math.argmax(labels, axis=1))
        total_predictions += images.shape[0]
    val_loss /= total_predictions
    val_acc = correct_predictions / total_predictions
    return val_loss, val_acc

# Dataset creation
datapath = 'C:/Users/lkang/Documents/ISIC_2019_Training_Input/'
csv_file_train = 'C:/Users/lkang/Documents/Master_Code_backup/Master_Code_backup/New UAP/ISIC2019_train.csv'
csv_file_val = 'C:/Users/lkang/Documents/Master_Code_backup/Master_Code_backup/New UAP/ISIC2019_test.csv'

train_dataset = ISICDatasetTF(datapath, csv_file_train, 'train').create_tf_dataset().batch(64).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = ISICDatasetTF(datapath, csv_file_val, 'val').create_tf_dataset().batch(64).prefetch(tf.data.experimental.AUTOTUNE)

# Main training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_dataset, criterion, optimizer)