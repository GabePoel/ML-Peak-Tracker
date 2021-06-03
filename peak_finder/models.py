"""
Various pre-made machine learning models made with the built in `peak_finder`
tools and provided by default.
"""

from . import utilities as util
from tensorflow.keras.models import load_model
import os


__all__ = [
    'simple_lorentzian',
    'tight_lorentzian',
    'wide_lorentzian',
    'split_lorentzian',
    'import_model']


def _parent(fp):
    """
    Parameters
    ----------
    fp : str
        File path.

    Returns
    -------
    str
        File path of parent.
    """
    return os.path.abspath(os.path.join(fp, os.pardir))


_trained_model_fp = os.path.join(
    _parent(os.path.abspath(__file__)), 'trained_models')


def simple_lorentzian():
    """
    A basic model detecting Lorentzians on flat data with little noise.
    """
    try:
        model = load_model(os.path.join(_trained_model_fp, 'simple_class'))
    except BaseException:
        model = load_model(
            os.path.join(_trained_model_fp, 'simple_class_backup'))
    return model


def tight_lorentzian():
    """
    A model detecting Lorentzians on noisy and non-flat data. But, only finds
    them if the window is very tightly centered. Often misses narrow peaks.
    """
    try:
        model = load_model(os.path.join(_trained_model_fp, 'tight_wiggle'))
    except BaseException:
        model = load_model(
            os.path.join(_trained_model_fp, 'tight_wiggle_backup'))
    return model


def wide_lorentzian():
    """
    A robust model that detects Lorentzians on noisy and non-flat data. But,
    often has a very wide area around the Lorentzians that might then need
    futher processing.
    """
    try:
        model = load_model(os.path.join(_trained_model_fp, 'wide_wiggle'))
    except BaseException:
        model = load_model(
            os.path.join(_trained_model_fp, 'wide_wiggle_backup'))
    return model


def split_lorentzian():
    """
    A model that tells whether there are one or two Lorentzians in the given
    input.
    """
    try:
        model = load_model(os.path.join(_trained_model_fp, 'simple_count'))
    except BaseException:
        model = load_model(
            os.path.join(_trained_model_fp, 'simple_count_backup'))
    return model


def import_model(path=None):
    """
    Import a model made with peak_finder but not provided here.
    """
    if path is None:
        model = load_model(util.load_file(path=path))
    return model
