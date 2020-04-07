import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import generate_data as gd
import classify_data as cd
import os

root = os.getcwd()
train_dir = os.path.join(root, 'generated_data_train')
test_dir = os.path.join(root, 'generated_data_test')

gd.generate_data_set(root, 'generated_data_train')
gd.generate_data_set(root, 'generated_data_test')

train_data_set = gd.load_data_set(train_dir)
test_data_set = gd.load_data_set(test_dir)

train_normalized = cd.normalize_data(train_data_set, no_f=True)
test_normalized = cd.normalize_data(test_data_set, no_f=True)

train_lorentz = train_normalized[0]
test_lorentz = test_normalized[0]

train_data = train_normalized[1]
test_data = test_normalized[1]

train_clusters = []
test_clusters = []
train_ranges = []
test_ranges = []

for i in range(0, len(train_data_set[1])):
    labels = cd.disect_lorentz_params_array(train_lorentz[i])
    train_clusters.append(labels[0])
    train_ranges.append(labels[1])
for i in range(0, len(test_data_set[1])):
    labels = cd.disect_lorentz_params_array(test_lorentz[i])
    test_clusters.append(labels[0])
    test_ranges.append(labels[1])

model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(32)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])