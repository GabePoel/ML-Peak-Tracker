"""
Framework for passive model training scripts.
"""

from . import utilities as util
from . import efficient_data_generation as ed
from . import classify_data as cd
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def set_default(parameter, default, num_models=1):
    """
    Set the default set of parameters to train models off of. Will return the
    provided parameters unless the array is empty. In that case, will provide
    the `default` paramters instead.

    Parameters
    ----------
    parameter : arr
        Lorentzian array.
    default : arr
        Lorentzian array.
    num_models : int, optional
        By default 1.

    Returns
    -------
    [type]
        [description]
    """
    if len(parameter) == 0:
        for i in range(0, num_models):
            parameter.append(default)
    return parameter


def passive_train(
        name='unnamed_model',
        location=None,
        data_size=10000,
        scale=(0, 1, 1024),
        expansion=2,
        noise=True,
        epochs=1000,
        overwrite=False,
        model_design=None,
        optimizer='Adadelta',
        loss=None,
        metrics=['accuracy'],
        stop_condition=False,
        steps=1,
        wiggle=0,
        verbose=1,
        min_noise_amp=1,
        max_noise_amp=1,
        min_noise_width=1,
        max_noise_width=1,
        no_quit=False,
        progress=True,
        backup=True,
        start_n=0,
        max_n=np.inf,
        split=False):
    """
    Passively trains a model that detects whether or not a Lorentzian is
    present.

    Parameters
    ----------
    name : str, optional
        Model name, by default 'unnamed_model'.
    location : str, optional
        Directory to save the model in, by default None.
    data_size : int, optional
        Size of data set to use for a batch of training, by default 10000.
    scale : tuple, optional
        Three element tuple that defines the scale of data that the model will
        be looking at, by default (0, 1, 1024). The zeroth element is the
        minimum value of the Lorentzian. The first value is the maximum. And
        the second value is the number of elements in the array. This is the
        default scale throughout `peak_finder` and is standard for normalized
        Lorentzians.
    expansion : int, optional
        How many times the full width at half maximum to extend data outside
        the Lorentzian, by default 2.
    noise : bool, optional
        Whether or not Gaussian noise should be added to the trainign data, by
        default True.
    epochs : int, optional
        Epochs, by default 1000.
    overwrite : bool, optional
        Overwrite existing model in save location, by default False.
    model_design : keras model, optional
        By default None. This sets up a three layer Sequential model.
    optimizer : str, optional
        By default 'Adadelta'.
    loss : keras loss function, optional
        by default None. This sets up a sparse categorical crossentropy loss.
    metrics : list, optional
        By default ['accuracy'].
    stop_condition : bool, optional
        Whether or not the training should stop on its own, by default False.
    steps : int, optional
        By default 1.
    wiggle : int, optional
        How many times the full width at half maximum to shift the center of
        the Lorentzian by, defaults to 0.
    verbose : int, optional
        By default 1.
    min_noise_amp : int, optional
        By default 1.
    max_noise_amp : int, optional
        By default 1.
    min_noise_width : int, optional
        by default 1.
    max_noise_width : int, optional
        By default 1.
    no_quit : bool, optional
        By default False.
    progress : bool, optional
        By default True.
    backup : bool, optional
        By default True.
    start_n : int, optional
        By default 0.
    max_n : [type], optional
        By default np.inf.
    split : bool, optional
        By default False.
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
        except BaseException:
            print('Latest model was corrupted. Loading backup model instead.')
            model = tf.keras.models.load_model(backup_path)
    if loss is None:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    n = start_n

    def train_routine():
        """
        Custom train routine based on the passed parameters to the parent
        function.
        """
        try:
            model.save(backup_path)
        except BaseException:
            pass
        print('\nStarting round ' + str(n))
        if not split:
            train = ed.make_single_data_set(
                number=data_size,
                scale=scale,
                expansion=expansion,
                noise=noise,
                wiggle=wiggle,
                min_noise_amp=min_noise_amp,
                max_noise_amp=max_noise_amp,
                min_noise_width=min_noise_width,
                max_noise_width=max_noise_width,
                progress=progress)
        else:
            train = ed.make_split_data_set(
                number=data_size,
                scale=scale,
                expansion=expansion,
                noise=noise,
                wiggle=wiggle,
                min_noise_amp=min_noise_amp,
                max_noise_amp=max_noise_amp,
                min_noise_width=min_noise_width,
                max_noise_width=max_noise_width,
                progress=progress)
        for i in range(0, steps):
            try:
                model.save(backup_path)
            except BaseException:
                pass
            model.fit(
                train[1],
                train[0],
                epochs=epochs,
                verbose=verbose,
                steps_per_epoch=data_size,
                shuffle=True)
            model.save(path)
            if progress:
                print('Done with step ' + str(i + 1) + ' of ' +
                      str(steps) + ' for round ' + str(n))
        if progress:
            print('Done with round ' + str(n))
    print('\n---------- Setup Complete ----------\n')
    if no_quit:
        while not stop_condition and n < max_n:
            try:
                n += 1
                train_routine()
            except BaseException:
                n -= 1
                print('An error occured. Restarting round.')

    else:
        while not stop_condition:
            n += 1
            train_routine()


def passive_class_train(
        name='unnamed_model',
        location=None,
        data_size=10000,
        scale=(
            0,
            1,
            1024),
        noise=True,
        epochs=1000,
        overwrite=False,
        model_design=None,
        optimizer='Adadelta',
        loss=None,
        metrics=['accuracy'],
        stop_condition=False,
        steps=1,
        verbose=1,
        no_quit=False):
    """
    Passively trains a model that classifies a given Lorentzian cluster as having some number of Lorentzians inside it.

    Parameters
    ----------
    name : str, optional
        Model name, by default 'unnamed_model'.
    location : str, optional
        Directory to save the model in, by default None.
    data_size : int, optional
        Size of data set to use for a batch of training, by default 10000.
    scale : tuple, optional
        Three element tuple that defines the scale of data that the model will
        be looking at, by default (0, 1, 1024). The zeroth element is the
        minimum value of the Lorentzian. The first value is the maximum. And
        the second value is the number of elements in the array. This is the
        default scale throughout `peak_finder` and is standard for normalized
        Lorentzians.
    noise : bool, optional
        Whether or not Gaussian noise should be added to the trainign data, by
        default True.
    epochs : int, optional
        Epochs, by default 1000.
    overwrite : bool, optional
        Overwrite existing model in save location, by default False.
    model_design : keras model, optional
        By default None. This sets up a three layer Sequential model.
    optimizer : str, optional
        By default 'Adadelta'.
    loss : keras loss function, optional
        by default None. This sets up a sparse categorical crossentropy loss.
    metrics : list, optional
        By default ['accuracy'].
    stop_condition : bool, optional
        Whether or not the training should stop on its own, by default False.
    steps : int, optional
        By default 1.
    verbose : int, optional
        By default 1.
    no_quit : bool, optional
        By default False.
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
        except BaseException:
            print('\nLatest model was corrupted. Loading backup model instead.')
            model = tf.keras.models.load_model(backup_path)
    if loss is None:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    try:
        n = util.load(n_path)
    except BaseException:
        n = 0
    try:
        running_data = util.load(data_path)
        running_labels = util.load(labels_path)
        print('Existing data and labels successfully loaded.')
    except BaseException:
        running_data = np.empty((0, scale[2]))
        running_labels = np.empty((0,))
        print('No existing data and labels found.')

    def train_routine(running_data, running_labels):
        """
        Custom train routine based on the passed parameters to the parent
        function.
        """
        try:
            model.save(backup_path)
        except BaseException:
            pass
        print('\n---------- Starting round ' + str(n) + ' ----------\n')
        simp = ed.make_simple_data_set(
            number=data_size, scale=scale, noise=noise)
        block = ed.convert_simple_data_set(simp)
        labels, data = cd.pre_process_for_equal_classifying(block)
        print('Now training over new data.')
        for i in range(0, steps):
            try:
                model.save(backup_path)
            except BaseException:
                pass
            model.fit(data, labels, epochs=epochs, verbose=verbose)
            model.save(path)
            print('Done with step ' + str(i + 1) + ' of ' +
                  str(steps) + ' for round ' + str(n))
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
                running_data, running_labels = train_routine(
                    running_data, running_labels)
            except BaseException:
                n -= 1
                print('An error occured. Restarting round.')
    else:
        while not stop_condition:
            n += 1
            running_data, running_labels = train_routine(
                running_data, running_labels)


def last_n(arr, n=10000):
    """
    Safely give the last n elements of the array.

    Parameters
    ----------
    arr : arr
        Array.
    n : int, optional
        By default 10000.

    Returns
    -------
    arr
        The truncated array.
    """
    m = len(arr)
    n = min(n, m)
    if len(arr.shape) == 2:
        return arr[max(m - n, 0):n, :]
    else:
        return arr[max(m - n, 0):n]
