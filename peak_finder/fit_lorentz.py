import numpy as np
import PySimpleGUI as sg
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

def set_n_least_squares(f, v, n=1, noise_filter=0, delta_f=None):
    """
    A least squares fit using the specified number of Lorentzians.
    """
    if delta_f is None:
        delta_f = np.inf
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
        bounds[0][i + 2 * n] = -1.5 * (max_f - min_f)
        bounds[1][i + 2 * n] = 1.5 * (max_f - min_f)
        bounds[0][i + 3 * n] = -np.inf
        bounds[1][i + 3 * n] = np.inf
    bounds[0][-2] = -np.inf
    bounds[0][-1] = -np.inf
    bounds[1][-2] = np.inf
    bounds[1][-1] = np.inf
    bounds = (np.array(bounds[0]), np.array(bounds[1]))
    try:
        fit = least_squares(multi_lorentz_residual_fit, p_estimate, ftol=None, gtol=None, xtol=1e-15, args=(f, v, np.ones(len(f))), method='lm')
    except:
        fit = least_squares(multi_lorentz_residual_fit, p_estimate, ftol=None, gtol=None, xtol=1e-15, args=(f, v, np.ones(len(f))), bounds=bounds)
    return fit, check_bounds(fit.x, bounds, noise_filter=noise_filter, delta_f=delta_f)

def free_n_least_squares(f, v, max_n=3, noise_filter=0, force_fit=False):
    """
    A least squares fit with an unspecified number of Lorentzians.
    """
    delta_f = (max(f) - min(f)) / len(f)
    best_fit, initial_keep = set_n_least_squares(f, v, n=1, noise_filter=noise_filter, delta_f=delta_f)
    n = 1
    while n <= max_n:
        n += 1
        new_fit, keep = set_n_least_squares(f, v, n, noise_filter=noise_filter)
        if util.order_difference(best_fit.cost, new_fit.cost) >= 1 and keep:
            best_fit = new_fit
            initial_keep = keep
        else:
            n = max_n + 1
    if initial_keep or force_fit:
        return best_fit
    else:
        return None

def fit_regions(f, v, regions, max_n=3, noise_filter=0):
    """
    Given frequency data, displacement data, and a regions array, will return a list of proposed Lorentzian parameters.
    """
    p_list = []
    for i in util.progressbar(range(0, len(regions)), prefix="Fitting: "):
        region_f, region_v = util.extract_region(i, regions, f, v)
        fit = free_n_least_squares(region_f, region_v, noise_filter=noise_filter)
        if fit is not None:
            p_list.append(free_n_least_squares(region_f, region_v, noise_filter=noise_filter).x)
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
    return p_table[p_table[:,1].argsort()]

def parameters_from_regions(f, v, regions, max_n=3, noise_filter=0, catch_degeneracies=True):
    """
    Given regions for analysis, frequency, displacement, will return parameters for the fit Lorentzians.
    """
    if len(regions) == 0:
        return np.empty((0, 4))
    p_list = fit_regions(f, v, regions, max_n=max_n, noise_filter=noise_filter)
    if len(p_list) == 0:
        return np.empty((0, 4))
    p_table = extract_parameters(p_list, noise_filter=noise_filter)
    if catch_degeneracies:
        p_table = remove_degeneracies(p_table, f)
    return correct_parameters(p_table)

def correct_parameters(p_table):
    """
    Puts Lorentzians into a somewhat more standard and consistent form.
    """
    parameters = np.empty((0, 4))
    for i in range(0, len(p_table)):
        p = p_table[i]
        if p[2] < 0:
            A = p[0]
            f0 = p[1]
            FWHM = -p[2]
            phase = np.pi - p[3]
            p = np.array([A, f0, FWHM, phase])
        if p[0] < 0:
            A = -p[0]
            f0 = p[1]
            FWHM = p[2]
            phase = p[3] + np.pi
            p = np.array([A, f0, FWHM, phase])
        if p[3] > 2 * np.pi or p[3] < 0:
            A = p[0]
            f0 = p[1]
            FWHM = p[2]
            phase = np.mod(p[3], 2 * np.pi)
            p = np.array([A, f0, FWHM, phase])
        parameters = np.append(parameters, np.array([p]), axis=0)
    return parameters
    
def regions_from_parameters(f, p):
    regions = np.empty((0, 2))
    for i in range(0, len(p)):
        f_min = p[i][1] - 2 * p[i][2]
        f_max = p[i][1] + 2 * p[i][2]
        ind_min = util.find_nearest_index(f, f_min)
        ind_max = util.find_nearest_index(f, f_max)
        region = np.array([[ind_min, ind_max]])
        regions = np.append(regions, region, axis=0)
    return regions

def check_bounds(p, bounds, noise_filter=0, delta_f=None):
    """
    Determines if the provided parameters violate the provided bounds.
    """
    if delta_f is None:
        delta_f = np.inf
    keep = True
    for i in range(0, len(p)):
        # Check if violates default bounds.
        if p[i] < bounds[0][i] or p[i] > bounds[1][i]:
            keep = False
    for i in range(0, len(p) // 4):
        # Check if A less than noise level.
        if keep:
            if np.abs(p[i]) < noise_filter:
                keep = False
    for i in range(2 * (len(p) // 4), 3 * (len(p) // 4)):
        # Check if FWHM less than distance between frequencies.
        if keep:
            if np.abs(p[i]) < delta_f:
                keep = False
    return keep

def lorentz_to_data(p, f, v=None, expansion=2):
    """
    Input: One row of extracted (final) parameters, f data, and v data if you want to approximate offset.
    Output: The f and v data of the fit Lorentzian within twice the FWHM.
    """
    offset = 0
    min_f_val = p[1] - expansion * np.abs(p[2])
    max_f_val = p[1] + expansion * np.abs(p[2])
    min_f_ind = util.find_nearest_index(f, min_f_val)
    max_f_ind = util.find_nearest_index(f, max_f_val)
    if v is not None:
        offset = np.mean(v[min_f_ind:max_f_ind])
    v = lorentz_fit(p, f)
    return_f = f[min_f_ind:max_f_ind]
    return_v = v[min_f_ind:max_f_ind] + offset
    return return_f, return_v

def lorentz_bounds_to_data(p, f, v, expansion=2):
    """
    Input: One row of extracted (final) parameters, f data, and v data.
    Output: The f and v data of the original data within twice the FWHM.
    """
    min_f_val = p[1] - expansion * np.abs(p[2])
    max_f_val = p[1] + expansion * np.abs(p[2])
    min_f_ind = util.find_nearest_index(f, min_f_val)
    max_f_ind = util.find_nearest_index(f, max_f_val)
    return_f = f[min_f_ind:max_f_ind]
    return_v = v[min_f_ind:max_f_ind]
    return return_f, return_v

def full_lorentz_to_data(p, f, expansion=2):
    """
    Input: One array from the original p_list, f data.
    Output: The f and v data of the fit Lorentzian within twice the FWHM.
    """
    n = len(p) // 4
    f0_arr = p[n:2 * n]
    FWHM_arr = p[2 * n:3 * n]
    min_f_val = min(f0_arr) - expansion * np.abs(max(FWHM_arr))
    max_f_val = max(f0_arr) + expansion * np.abs(max(FWHM_arr))
    min_f_ind = util.find_nearest_index(f, min_f_val)
    max_f_ind = util.find_nearest_index(f, max_f_val)
    v = multi_lorentz_fit(p, f)
    return_f = f[min_f_ind:max_f_ind]
    return_v = v[min_f_ind:max_f_ind]
    return return_f, return_v

def full_lorentz_bounds_to_data(p, f, v, expansion=2):
    """
    Input: One array from the original p_list, f data.
    Output: The f and v data of the original data within twice the FWHM.
    """
    n = len(p) // 4
    f0_arr = p[n:2 * n]
    FWHM_arr = p[2 * n:3 * n]
    min_f_val = min(f0_arr) - expansion * np.abs(max(FWHM_arr))
    max_f_val = max(f0_arr) + expansion * np.abs(max(FWHM_arr))
    min_f_ind = util.find_nearest_index(f, min_f_val)
    max_f_ind = util.find_nearest_index(f, max_f_val)
    return_f = f[min_f_ind:max_f_ind]
    return_v = v[min_f_ind:max_f_ind]
    return return_f, return_v

def catch_degeneracies(p1, p2, f, allowed_delta_ind=10):
    """
    Input: Two elements of p_table and f data.
    Output: Whether or not they are fitting to the same Lorentzian. True if they are. False if they aren't.
    """
    delta_ind = util.compare_lorentz(p1, p2, f)
    degenerate = delta_ind < allowed_delta_ind
    return degenerate

def remove_degeneracies(p_table, f, allowed_delta_ind=10):
    """
    Removes duplicate Lorentzians.
    """
    degeneracy_table = []
    new_p_tabel = np.empty((0, 4))
    for i in util.progressbar(range(0, len(p_table)), prefix="Checking for Degeneracies: "):
        degeneracy_table.append([])
        for j in range(0, len(p_table)):
            if catch_degeneracies(p_table[i], p_table[j], f, allowed_delta_ind=allowed_delta_ind):
                degeneracy_table[i].append(j)
    for i in range(0, len(degeneracy_table)):
        if len(degeneracy_table[i]) > 1:
            for k in range(1, len(degeneracy_table[i])):
                if len(degeneracy_table[i]) > 1:
                    j = degeneracy_table[i][k]
                    degeneracy_table[j] = []
    for i in range(0, len(degeneracy_table)):
        if len(degeneracy_table[i]) > 0:
            new_p_tabel = np.append(new_p_tabel, np.array([p_table[i]]), axis=0)
    return new_p_tabel[new_p_tabel[:,1].argsort()]


def parameters_from_selections(data_files, region_selections):
    sg.one_line_progress_meter_cancel('-key-')
    all_peaks = []
    counter = 0
    for i in util.progressbar(range(0, len(data_files)), "Fitting: ", progress=False):
        all_peaks.append(np.empty((0, 4)))
        for j in range(0, len(region_selections)):
            try:
                sg.one_line_progress_meter('Overall Fitting Progress', counter, len(data_files) * len(region_selections), '-key-')
                f = data_files[i].f
                v = data_files[i].r
                regions = region_selections[j][i]
                params = parameters_from_regions(f, v, regions)
                if len(params) == 0:
                    params = np.array([[np.nan, np.nan, np.nan, np.nan]])
            except:
                params = np.array([[np.nan, np.nan, np.nan, np.nan]])
            all_peaks[i] = np.append(all_peaks[i], params, axis=0)
            counter += 1
    sg.one_line_progress_meter_cancel('-key-')
    return np.array(all_peaks)