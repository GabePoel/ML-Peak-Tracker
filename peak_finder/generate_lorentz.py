"""
Generates Lorentzians and provides functional forms for fitting.
"""

from . import classify_data as cd
from . import utilities as util
import numpy as np


def in_phase_lorentz(A, f0, FWHM, f):
    """
    Out of phase Lorentzian with amplitude A centered at f0.
    """
    return (A / (2 * FWHM)) * (f - f0) / (((f - f0) / FWHM) ** 2 + 1 / 4)


def out_phase_lorentz(A, f0, FWHM, f):
    """
    In phase Lorentzian with amplitude A centered at f0.
    """
    return (A / 4) * (1 / (((f - f0) / FWHM) ** 2 + 1 / 4))


def complex_lorentz(A, f0, FWHM, f, phase):
    """
    Arbitrary phase Lorentzian with amplitude A centered at f0.
    """
    return np.cos(phase) * in_phase_lorentz(A, f0, FWHM, f) + \
        np.sin(phase) * out_phase_lorentz(A, f0, FWHM, f)


def generate_lorentz(f):
    """
    Returns displacement and parameters for one random Lorentzian
    """
    f_distance = max(f) - min(f)
    f0 = f_distance * np.random.random() + min(f)
    phase = 2 * np.pi * np.random.random()
    A = (100 * (np.random.random() + 0.3))
    FWHM = f_distance * (np.random.random() + 0.05) / 50
    params = np.array([[A, f0, FWHM, phase]])
    return complex_lorentz(A, f0, FWHM, f, phase), params


def generate_multi_lorentz(f, n):
    """
    Returns displacement and parameters for n random Lorentzians.
    """
    displacement = 0
    params = np.empty((0, 4))
    for i in range(0, n):
        disp_addition, params_addition = generate_lorentz(f)
        displacement += disp_addition
        params = np.append(params, params_addition, axis=0)
    params = params[params[:, 1].argsort()]
    return displacement, params


def generate_background(f):
    """
    Returns one really big background Lorentzian and its paramteters.
    """
    f_distance = 100 * (max(f) - min(f))
    f0 = f_distance * np.random.random() + (100 * min(f))
    phase = 2 * np.pi * np.random.random()
    A = 5000 * np.random.random()
    FWHM = f_distance * np.random.random()
    params = np.array([[A, f0, FWHM, phase]])
    return complex_lorentz(A, f0, FWHM, f, phase), params


def generate_noise(f, amount=1, width=1):
    """
    Returns slight background noise.
    """
    sample = np.random.normal(0, amount, f.shape)
    reduced_sample = sample[0:int(np.round(len(sample) / width))]
    reduced_sample = cd.normalize_1d(
        reduced_sample, scale=(min(sample), max(sample), len(sample)))
    return reduced_sample


def generate_normalized_data(
    include_noise=True,
    max_num_lorentz=16,
    scale=(0, 1, 1024)):
    """
    Generates data that's already pre-normalized to the provided scale.
    """
    bg, l, f, v = generate_data(
        include_noise=include_noise, max_num_lorentz=max_num_lorentz)
    bg, l, f, v = cd.normalize_data(bg, l, f, v, scale=scale)
    return bg, l, f, v


def generate_data(include_noise=True, max_num_lorentz=16):
    """
    Returns randomly generated data set and all its defining parameters.
    """
    if max_num_lorentz < 1:
        num_lorentz = 0
    elif max_num_lorentz == 1:
        num_lorentz = 1
    else:
        num_lorentz = np.random.randint(max_num_lorentz - 1) + 1
    f1, f2 = 0, 0
    while f1 == f2:
        f1 = np.random.random() * 4000 - 4000
        f2 = np.random.random() * 4000
    f_min, f_max = min(f1, f2), max(f1, f2)
    f = np.linspace(f_min, f_max, int(2 * (f_max - f_min)))
    noise = generate_noise(f)
    if not include_noise:
        noise *= 0
    background, background_params = generate_background(f)
    lorentz, lorentz_params = generate_multi_lorentz(f, num_lorentz)
    displacement = noise + background + lorentz
    return background_params, lorentz_params, f, displacement


def multi_lorentz(f, A_arr, f0_arr, FWHM_arr, phase_arr):
    """
    Generates a combined sum of numerous Lorentzians.
    """
    return np.sum(
        complex_lorentz(
            A_arr,
            f0_arr,
            FWHM_arr,
            f,
            phase_arr),
        axis=0)


def multi_lorentz_2d(f, params):
    """
    Generates a combined sum of numerous Lorentzians given a single 2D array of
    parameters.
    """
    params = util.remove_nans(params)
    A_arr = params[:, [0]]
    f0_arr = params[:, [1]]
    FWHM_arr = params[:, [2]]
    phase_arr = params[:, [3]]
    return multi_lorentz(f, A_arr, f0_arr, FWHM_arr, phase_arr)
