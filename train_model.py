import tensorflow as tf
import numpy as np
import efficient_data_generation as ed
import classify_data as cd
import os

# This is the framework for the passive model training scripts.

def passive_train(name='unnamed_model', location=None, data_size=10000, scale=(0,1,1024), expansion=2, noise=True, epochs=1000, overwrite=False, model_design=None, optimizer='Adadelta', loss=None, metrics=['accuracy'], stop_condition=False, steps=1, wiggle=0, verbose=1):
    path = os.path.join(location, name)
    if model_design is None:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(scale[2], activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
    else:
        model = model_design
    if os.path.exists(path) and not overwrite:
        model = tf.keras.models.load_model(path)
    if loss is None:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    n = 0
    while not stop_condition:
        n += 1
        print('Starting round ' + str(n))
        train = ed.make_single_data_set(number=data_size, scale=scale, expansion=expansion, noise=noise, wiggle=wiggle)
        for i in range(0, steps):
            model.fit(train[1], train[0], epochs=epochs, verbose=verbose)
            model.save(path)
            print('Done with step ' + str(i + 1) + ' of ' + str(steps) + ' for round ' + str(n))
        print('Done with round ' + str(n))

def passive_class_train(name='unnamed_model', location=None, data_size=10000, scale=(0,1,1024), noise=True, epochs=1000, overwrite=False, model_design=None, optimizer='Adadelta', loss=None, metrics=['accuracy'], stop_condition=False, steps=1, verbose=1):
    path = os.path.join(location, name)
    if model_design is None:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(scale[2], activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
    else:
        model = model_design
    if os.path.exists(path) and not overwrite:
        model = tf.keras.models.load_model(path)
    if loss is None:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    n = 0
    while not stop_condition:
        n += 1
        print('Starting round ' + str(n))
        simp = ed.make_simple_data_set(number=data_size, scale=scale, noise=noise)
        block = ed.convert_simple_data_set(simp)
        labels, data = cd.pre_process_for_equal_classifying(block)
        for i in range(0, steps):
            model.fit(data, labels, epochs=epochs, verbose=verbose)
            model.save(path)
            print('Done with step ' + str(i + 1) + ' of ' + str(steps) + ' for round ' + str(n))
        print('Done with round ' + str(n))