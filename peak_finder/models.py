import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import os
try:
    from . import utilities as util
except:
    import utilities as util

def simple_lorentzian():
    """
    A basic model detecting Lorentzians on flat data with little noise.
    """
    try:
        model = load_model(os.path.join(os.getcwd(), 'peak_finder', 'trained_models',  'simple_class'))
    except:
        model = load_model(os.path.join(os.getcwd(), 'peak_finder', 'trained_models',  'simple_class_backup'))
    return model

def tight_lorentzian():
    """
    A model detecting Lorentzians on noisy and non-flat data. But, only finds them if the window is very tightly centered. Often misses narrow peaks.
    """
    try:
        model = load_model(os.path.join(os.getcwd(), 'peak_finder', 'trained_models',  'tight_wiggle'))
    except:
        model = load_model(os.path.join(os.getcwd(), 'peak_finder', 'trained_models',  'tight_wiggle_backup'))
    return model

def wide_lorentzian():
    """
    A robust model that detects Lorentzians on noisy and non-flat data. But, often has a very wide area around the Lorentzians that might then need futher processing.
    """
    try:
        model = load_model(os.path.join(os.getcwd(), 'peak_finder', 'trained_models',  'wide_wiggle'))
    except:
        model = load_model(os.path.join(os.getcwd(), 'peak_finder', 'trained_models',  'wide_wiggle_backup'))
    return model

def split_lorentzian():
    """
    A model that tells whether there are one or two Lorentzians in the given input.
    """
    try:
        model = load_model(os.path.join(os.getcwd(), 'peak_finder', 'trained_models', 'simple_count'))
    except:
        model = load_model(os.path.join(os.getcwd(), 'peak_finder', 'trained_models', 'simple_count_backup'))
    return model

def import_model(path=None):
    if path is None:
        model = load_model(util.load_file(path=path))
    return model