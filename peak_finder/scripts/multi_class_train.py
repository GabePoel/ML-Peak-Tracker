import tensorflow as tf
import os
import sys
root = os.getcwd()
parent = os.path.join(root, '..')
sys.path.insert(1, parent)
import train_model as tm

# SYLVIA DON'T RUN THIS ONE, I'M NOT USING IT ATM
# Trains several models.
# This model says whether an input is a Lorentzian or not.
# It is only trained on Lorentzians exactly centered within the data tightly framed around them.

default_location = os.path.join(parent, 'models')
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

name=['simple_class', 'tight_wiggle', 'wide_wiggle']
location=[default_location, default_location, default_location]
data_size=[1000, 1000, 1000]
scale=[(0,1,1024), (0,1,1024), (0,1,1024)]
expansion=[1.5, 1, 1]
noise=[True, True, True]
epochs=[1, 1, 1]
overwrite=[False, False, False]
model_design=[model_1, model_2, model_3]
optimizer=['Adadelta', 'Adadelta', 'Adadelta']
loss=[None, None, None]
metrics=[['Accuracy'], ['Accuracy'], ['Accuracy']]
stop_condition=[False, False, False]
steps=[1, 1, 1]
wiggle=[0, 2, 10]
verbose=[1, 1, 1]
min_noise_amp=[1, 0.01, 0.01]
max_noise_amp=[1, 5, 5]
min_noise_width=[1, 1, 1]
max_noise_width=[1, 19, 19]
no_quit=[True, True, True]
progress=[True, True, True]
backup=[True, True, True]

tm.passive_multi_train(3, name, location, data_size, scale, expansion, noise, epochs, overwrite, model_design, optimizer, loss, metrics, stop_condition, steps, wiggle, verbose, min_noise_amp, max_noise_amp, min_noise_width, max_noise_width, no_quit, progress, backup)