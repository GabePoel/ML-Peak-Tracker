import tensorflow as tf
import os
import sys
root = os.getcwd()
parent = os.path.join(root, '..')
sys.path.insert(1, parent)
import train_model as tm

# SYLVIA RUN THIS SCRIPT PLS THX
# Trains the count_hidden_layer_1024 model.
# This is the model that 'counts' how many Lorentzians are in a givin region that supposedly has Lorentzians.
# It only understands the numbers 1 through 4, so don't be too harsh on it.
# It's doing its best.

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax')
])

root = os.getcwd()
location = os.path.join(parent, 'models')
tm.passive_class_train(name='simple_count', location=location, data_size=5000, scale=(0,1,1024), noise=True, epochs=100, model_design=model, steps=10, optimizer='adadelta')
