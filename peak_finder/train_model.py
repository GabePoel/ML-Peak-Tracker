import tensorflow as tf
import numpy as np
import os
from . import efficient_data_generation as ed
from . import classify_data as cd

# This is the framework for the passive model training scripts.

def set_default(parameter, default, num_models=1):
    if len(parameter) == 0:
        for i in range(0, num_models):
            parameter.append(default)
    return parameter

def passive_train(name='unnamed_model', location=None, data_size=10000, scale=(0,1,1024), expansion=2, noise=True, epochs=1000, overwrite=False, model_design=None, optimizer='Adadelta', loss=None, metrics=['accuracy'], stop_condition=False, steps=1, wiggle=0, verbose=1, min_noise_amp=1, max_noise_amp=1, min_noise_width=1, max_noise_width=1, no_quit=False, progress=True, backup=True, start_n=0, max_n=np.inf):
    """
    Passively trains a model that detects whether or not a Lorentzian is present.
    """
    path = os.path.join(location, name)
    backup_path = os.path.join(location, name + '_backup')
    if model_design is None:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(scale[2], activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
    else:
        model = model_design
    if os.path.exists(path) and not overwrite:
        try:
            model = tf.keras.models.load_model(path)
        except:
            print('Latest model was corrupted. Loading backup model instead.')
            model = tf.keras.models.load_model(backup_path)
    if loss is None:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    n = start_n
    def train_routine():
        try:
            model.save(backup_path)
        except:
            pass
        print('\nStarting round ' + str(n))
        train = ed.make_single_data_set(number=data_size, scale=scale, expansion=expansion, noise=noise, wiggle=wiggle, min_noise_amp=min_noise_amp, max_noise_amp=max_noise_amp, min_noise_width=min_noise_width, max_noise_width=max_noise_width, progress=progress)
        for i in range(0, steps):
            try:
                model.save(backup_path)
            except:
                pass
            model.fit(train[1], train[0], epochs=epochs, verbose=verbose, steps_per_epoch=data_size, shuffle=True)
            model.save(path)
            if progress:
                print('Done with step ' + str(i + 1) + ' of ' + str(steps) + ' for round ' + str(n))
        if progress:
            print('Done with round ' + str(n))
    print('\n---------- Setup Complete ----------\n')
    if no_quit:
        while not stop_condition and n < max_n:
            try:
                n += 1
                train_routine()
            except:
                n -= 1
                print('An error occured. Restarting round.')
                
    else:
        while not stop_condition:
            n += 1
            train_routine()

def passive_class_train(name='unnamed_model', location=None, data_size=10000, scale=(0,1,1024), noise=True, epochs=1000, overwrite=False, model_design=None, optimizer='Adadelta', loss=None, metrics=['accuracy'], stop_condition=False, steps=1, verbose=1, no_quit=False):
    """
    Passively trains a model that classifies a given Lorentzian cluster as having some number of Lorentzians inside it.
    """
    path = os.path.join(location, name)
    backup_path = os.path.join(location, name + '_backup')
    if model_design is None:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(scale[2], activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
    else:
        model = model_design
    if os.path.exists(path) and not overwrite:
        try:
            model = tf.keras.models.load_model(path)
        except:
            print('Latest model was corrupted. Loading backup model instead.')
            model = tf.keras.models.load_model(backup_path)
    if loss is None:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    n = 0
    def train_routine():
        try:
            model.save(backup_path)
        except:
            pass
        print('Starting round ' + str(n))
        simp = ed.make_simple_data_set(number=data_size, scale=scale, noise=noise)
        block = ed.convert_simple_data_set(simp)
        labels, data = cd.pre_process_for_equal_classifying(block)
        for i in range(0, steps):
            try:
                model.save(backup_path)
            except:
                pass
            model.fit(data, labels, epochs=epochs, verbose=verbose)
            model.save(path)
            print('Done with step ' + str(i + 1) + ' of ' + str(steps) + ' for round ' + str(n))
        print('Done with round ' + str(n))
    print('\n---------- Setup Complete ----------\n')
    if no_quit:
        while not stop_condition:
            try:
                n += 1
                train_routine()
            except:
                n -= 1
                print('An error occured. Restarting round.')
    else:
        while not stop_condition:
            n += 1
            train_routine()
