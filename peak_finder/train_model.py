import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
try:
    from . import utilities as util
    from . import efficient_data_generation as ed
    from . import classify_data as cd
except:
    import utilities as util
    import efficient_data_generation as ed
    import classify_data as cd

# This is the framework for the passive model training scripts.

def set_default(parameter, default, num_models=1):
    if len(parameter) == 0:
        for i in range(0, num_models):
            parameter.append(default)
    return parameter

def passive_train(name='unnamed_model', location=None, data_size=10000, scale=(0,1,1024), expansion=2, noise=True, epochs=1000, overwrite=False, model_design=None, optimizer='Adadelta', loss=None, metrics=['accuracy'], stop_condition=False, steps=1, wiggle=0, verbose=1, min_noise_amp=1, max_noise_amp=1, min_noise_width=1, max_noise_width=1, no_quit=False, progress=True, backup=True, start_n=0, max_n=np.inf, split=False):
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
        if not split:
            train = ed.make_single_data_set(number=data_size, scale=scale, expansion=expansion, noise=noise, wiggle=wiggle, min_noise_amp=min_noise_amp, max_noise_amp=max_noise_amp, min_noise_width=min_noise_width, max_noise_width=max_noise_width, progress=progress)
        else:
            train = ed.make_split_data_set(number=data_size, scale=scale, expansion=expansion, noise=noise, wiggle=wiggle, min_noise_amp=min_noise_amp, max_noise_amp=max_noise_amp, min_noise_width=min_noise_width, max_noise_width=max_noise_width, progress=progress)
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
    data_path = os.path.join(location, name + '_data.pkl')
    labels_path = os.path.join(location, name + '_labels.pkl')
    n_path = os.path.join(location, name + '_n.pkl')
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
            print('\nModel successfully loaded.')
        except:
            print('\nLatest model was corrupted. Loading backup model instead.')
            model = tf.keras.models.load_model(backup_path)
    if loss is None:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    try:
        n = util.load(n_path)
    except:
        n = 0
    try:
        running_data = util.load(data_path)
        running_labels = util.load(labels_path)
        print('Existing data and labels successfully loaded.')
    except:
        running_data = np.empty((0, scale[2]))
        running_labels = np.empty((0,))
        print('No existing data and labels found.')
    def train_routine(running_data, running_labels):
        try:
            model.save(backup_path)
        except:
            pass
        print('\n---------- Starting round ' + str(n) + ' ----------\n')
        simp = ed.make_simple_data_set(number=data_size, scale=scale, noise=noise)
        block = ed.convert_simple_data_set(simp)
        labels, data = cd.pre_process_for_equal_classifying(block)
        print('Now training over new data.')
        for i in range(0, steps):
            try:
                model.save(backup_path)
            except:
                pass
            model.fit(data, labels, epochs=epochs, verbose=verbose)
            model.save(path)
            print('Done with step ' + str(i + 1) + ' of ' + str(steps) + ' for round ' + str(n))
        running_data = last_n(np.append(running_data, data, axis=0))
        running_labels = last_n(np.append(running_labels, labels, axis=0))
        print('Now training over old data.')
        model.fit(running_data, running_labels, verbose=verbose)
        model.save(path)
        util.save(running_data, data_path)
        util.save(running_labels, labels_path)
        print('Done with round ' + str(n))
        util.save(n, n_path)
        return running_data, running_labels
    print('\n---------- Setup Complete ----------\n')
    if no_quit:
        while not stop_condition:
            try:
                n += 1
                running_data, running_labels = train_routine(running_data, running_labels)
            except:
                n -= 1
                print('An error occured. Restarting round.')
    else:
        while not stop_condition:
            n += 1
            running_data, running_labels = train_routine(running_data, running_labels)

def last_n(arr, n=10000):
    m = len(arr)
    n = min(n, m)
    # print(arr)
    if len(arr.shape) == 2:
        return arr[max(m - n, 0):n, :]
    else:
        return arr[max(m - n, 0):n]
