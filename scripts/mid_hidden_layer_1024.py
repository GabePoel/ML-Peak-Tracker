import tensorflow as tf
import os
import sys
root = os.getcwd()
parent = os.path.join(root, '..')
sys.path.insert(1, parent)
import train_model as tm

# SYLVIA DON'T RUN THIS ONE, I'M NOT USING IT ATM
# Trains the mid_hidden_layer_1024 model.
# This model says whether an input is a Lorentzian or not.
# It is only trained on Lorentzians exactly centered within the data tightly framed around them.

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

location = os.path.join(parent, 'models')
tm.passive_train(name='mid_hidden_layer_1024', location=location, data_size=10000, scale=(0,1,1024), expansion=1.5, noise=True, epochs=10, model_design=model, steps=100)