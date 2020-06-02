import tensorflow as tf
import os
import sys
root = os.getcwd()
parent = os.path.join(root, '..')
sys.path.insert(1, parent)
import train_model as tm

# SYLVIA DON'T RUN THIS ONE, I'M DOING THAT
# Trains the wiggle_hidden_layer_1024 model.
# This model says whether an input is a Lorentzian or not.
# It is trained on Lorentzians that aren't exactly centered and have a little 'wiggle' room around them.

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

root = os.getcwd()
location = os.path.join(parent, 'models')
tm.passive_train(name='tight_wiggle', location=location, data_size=1000, scale=(0,1,1024), expansion=1, noise=True, epochs=1, model_design=model, steps=1, wiggle=2, min_noise_amp=0.01, max_noise_amp=5, min_noise_width=1, max_noise_width=10, no_quit=True, verbose=1, progress=True)