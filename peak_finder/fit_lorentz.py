import numpy as np
from scipy.optimize import least_squares
from . import utilities as util

def estimate_parameters(f, v, n=1, estimate_background=True):
    """
    Given some frequency and displacement data as well as how many Lorentzians are in it, will ballpark some approximate parameters.
    """
    max_f = max(f)
    min_f = min(f)
    max_v = max(v)
    min_v = min(v)
    guess_FWHM = np.ones((1, n)) * (max_f - min_f) / n
    guess_f0 = (np.arange(n).reshape((1, n)) + 1) * (max_f - min_f) / (n + 1) + min_f
    guess_A = np.ones((1, n)) * (max_v - min_v) / n
    guess_phase = np.ones((1, n)) * np.pi
    guess_params = np.concatenate([guess_A, guess_f0, guess_FWHM, guess_phase], axis=0)
    if estimate_background:
        background = np.array([[np.mean(v), 0]])
        guess_params = np.append(guess_params, background)
    guess_params = np.transpose(guess_params)
    return guess_params

def in_lorentz_fit(p, f):
    """
    An in phase Lorentzian for fitting.
    p = array([Amplitude, Center Frequency, Full Width at Half Maximum])
    """
    A = p[0]
    f0 = p[1]
    FWHM = p[2]
    return (A / (2 * FWHM)) * (f - f0) / (((f - f0) / FWHM) **2 + 1 / 4)

def out_lorentz_fit(p, f):
    """
    An out of phase Lorentzian for fitting.
    p = array([Amplitude, Center Frequency, Full Width at Half Maximum])
    """
    A = p[0]
    f0 = p[1]
    FWHM = p[2]
    return (A / 4) * (1 / (((f - f0) / FWHM) ** 2 + 1 / 4))

def lorentz_fit(p, f):
    """
    An arbitrary Lorentzian for fitting.
    p = array([Amplitude, Center Frequency, Full Width at Half Maximum, Phase])
    """
    p_no_phase = p[:3]
    return np.cos(p[3]) * in_lorentz_fit(p_no_phase, f) + np.sin(p[3]) * out_lorentz_fit(p_no_phase, f)

def lorentz_residual_fit(p, f, z):
    """
    As lorentz_fit, but with z for the residuals.
    """
    return lorentz_fit(p, f) - z

def multi_lorentz_fit(p, f):
    """
    An arbitrary number of Lorentzians for fitting.
    p = array([Amplitude 1, Amplitude 2, ..., Center Frequency 1, Center Frequency 2, ..., Full Width at Half Maximum 1, Full Width at Half Maximum 2, ..., Phase 1, Phase 2, ...])
    """
    slope = p[-1]
    offset = p[-2]
    p_no_background = p[:-2]
    n = int(len(p_no_background) / 4)
    result = slope * f + offset
    for i in range(0, n):
        A = p_no_background[i]
        f0 = p_no_background[i + n]
        FWHM = p_no_background[i + 2 * n]
        phase = p_no_background[i + 3 * n]
        p_single = np.array([A, f0, FWHM, phase])
        result = result + lorentz_fit(p_single, f)
    return result

def multi_lorentz_residual_fit(p, f, z, w):
    """
    As multi_lorentz_fit, but with z for the residuals.
    """
    return (multi_lorentz_fit(p, f) - z) * w

def set_n_least_squares(f, v, n=1):
    """
    A least squares fit using the specified number of Lorentzians.
    """
    p_estimate = estimate_parameters(f, v, n)
    bounds = ([None] * len(p_estimate), [None] * len(p_estimate))
    min_f = min(f)
    max_f = max(f)
    max_A = max(v) - min(v)
    for i in range(0, n):
        bounds[0][i] = -2 * max_A
        bounds[1][i] = 2 * max_A
        bounds[0][i + n] = min_f - (max_f - min_f)
        bounds[1][i + n] = max_f + (max_f - min_f)
        bounds[0][i + 2 * n] = -3 * (max_f - min_f)
        bounds[1][i + 2 * n] = 3 * (max_f - min_f)
        bounds[0][i + 3 * n] = -np.inf
        bounds[1][i + 3 * n] = np.inf
    bounds[0][-2] = -np.inf
    bounds[0][-1] = -np.inf
    bounds[1][-2] = np.inf
    bounds[1][-1] = np.inf
    bounds = (np.array(bounds[0]), np.array(bounds[1]))
    if len(p_estimate) <= len(f):
        fit = least_squares(multi_lorentz_residual_fit, p_estimate, ftol=None, gtol=None, xtol=1e-15, args=(f, v, np.ones(len(f))), method='lm')
    else:
        fit = least_squares(multi_lorentz_residual_fit, p_estimate, ftol=None, gtol=None, xtol=1e-15, args=(f, v, np.ones(len(f))), bounds=bounds)
    return fit, check_bounds(fit.x, bounds)

def free_n_least_squares(f, v, max_n=3):
    """
    A least squares fit with an unspecified number of Lorentzians.
    """
    best_fit, initial_keep = set_n_least_squares(f, v, n=1)
    n = 1
    while n <= max_n:
        n += 1
        new_fit, keep = set_n_least_squares(f, v, n)
        if util.order_difference(best_fit.cost, new_fit.cost) >= 1 and keep:
            best_fit = new_fit
        else:
            n = max_n + 1
    return best_fit

def fit_regions(f, v, regions, max_n=3):
    """
    Given frequency data, displacement data, and a regions array, will return a list of proposed Lorentzian parameters.
    """
    p_list = []
    for i in range(0, len(regions)):
        region_f, region_v = util.extract_region(i, regions, f, v)
        p_list.append(free_n_least_squares(region_f, region_v).x)
    return p_list

def extract_parameters(p_list, noise_filter=0):
    """
    Given a list of Lorentzian parameters and a noise level, will return a complete and usable numpy array of the parameters.
    """
    p_table = np.empty((0, 4))
    for i in range(0, len(p_list)):
        working_p = p_list[i][0:-2]
        working_p = working_p.reshape(4, len(working_p) // 4)
        working_p = np.transpose(working_p)
        for j in range(0, len(working_p)):
            if np.abs(working_p[j][0]) >= noise_filter:
                p_table = np.append(p_table, working_p, axis=0)
    return p_table

def check_bounds(p, bounds):
    """
    Determines if the provided parameters violate the provided bounds.
    """
    keep = True
    for i in range(0, len(p)):
        if p[i] < bounds[0][i] or p[i] > bounds[1][i]:
            keep = False
    return keep