import numpy as np
import PySimpleGUI as sg
import os
from . import fit_lorentz as fl
from . import sliding_window as sw
from . import utilities as util
from . import models
from . import automatic

def import_tdms_files(path=None):
    path = util.load_dir(path)
    names = os.listdir(path)
    data_files = []
    for name in names:
        if name[-5:] == '.tdms':
            stamp = name[:-5]
            file_path = os.path.join(path, name)
            data_file = util.import_file(file_path)
            data_file.import_meta(stamp)
            data_files.append(data_file)
    data_files.sort(key=lambda d: int(str(d.date) + str(d.time)))
    return data_files

def find_last_peak(params_3d, index):
    if not any(np.isnan(params_3d[-1][index])):
        return_params_1d = params_3d[-1][index]
    else:
        return_params_1d = find_last_peak(params_3d[0:-1], index)
    return return_params_1d

def get_reference_params(params_3d):
    reference_params_2d = np.empty((0, 4))
    for i in range(0, len(params_3d[0])):
        params_1d = find_last_peak(params_3d, i)
        reference_params_2d = np.append(reference_params_2d, np.array([params_1d]), axis=0)
    return reference_params_2d

def find_min_f(f, active_params, active_index):
    if active_index == 0:
        return min(f)
    elif not np.isnan(active_params[active_index - 1][1]):
        return active_params[active_index - 1][1] + active_params[active_index - 1][2]
    else:
        return find_min_f(f, active_params, active_index - 1)

def find_max_f(f, active_params, active_index):
    if active_index == len(active_params) - 1:
        return max(f)
    elif not np.isnan(active_params[active_index + 1][1]):
        return active_params[active_index + 1][1] - active_params[active_index + 1][2]
    else:
        return find_max_f(f, active_params, active_index + 1)

def find_extremes_f(f, active_params, active_index):
    f1 = find_min_f(f, active_params, active_index)
    f2 = find_max_f(f, active_params, active_index)
    min_f = min(f1, f2)
    max_f = max(f1, f2)
    return min_f, max_f

def find_missing_peaks(f, v, active_params, reference_params):
    updated_params = np.empty((0, 4))
    noise_level = 3 * sw.extract_noise(f)
    model = models.tight_lorentzian()
    for i in range(0, len(active_params)):
        if not any(np.isnan(active_params[i])):
            updated_params = np.append(updated_params, np.array([active_params[i]]), axis=0)
        else:
            min_f, max_f = find_extremes_f(f, active_params, i)
            min_ind = util.find_nearest_index(f, min_f)
            max_ind = util.find_nearest_index(f, max_f)
            reference_FWHM = reference_params[i][2]
            delta_f = max_f - min_f
            try:
                base_zoom = int(np.round(np.log2(delta_f / reference_FWHM)))
            except:
                print(reference_params[i])
                print(delta_f)
                print(min_f)
                print(max_f)
            # make sure we don't waste computing resources looking at nothing
            if base_zoom > 8:
                base_zoom = 8
            regions = np.array([[min_ind, max_ind]])
            potential_regions = sw.split_peaks(model, f, v, regions, base_zoom - 1, base_zoom + 1)
            if len(potential_regions) > 0:
                potential_params = fl.parameters_from_regions(f, v, potential_regions, noise_filter=noise_level)
                if len(potential_params) > 0:
                    f0_vals = potential_params[:,1]
                    reference_f0 = reference_params[i][1]
                    closest_index = util.find_nearest_index(f0_vals, reference_f0)
                    closest_params = potential_params[closest_index]
                    updated_params = np.append(updated_params, np.array([closest_params]), axis=0)
                else:
                    updated_params = np.append(updated_params, np.array([[np.nan, np.nan, np.nan, np.nan]]), axis=0)
            else:
                updated_params = np.append(updated_params, np.array([[np.nan, np.nan, np.nan, np.nan]]), axis=0)
    return updated_params

def track_temperatures(data_files, initial_params=None, show=True, depth=None, learn=True, correction=True):
    # previous refers to the index before
    # reference refers to the last known usable values (could be index before or earlier)
    # active refers to the current index
    sg.one_line_progress_meter_cancel('-key-')
    if initial_params is None:
        initial_params = np.empty((0, 4))
        initial_f = data_files[0].f
        initial_v = data_files[0].r
        initial_params = automatic.quick_analyze(initial_f, initial_v, show=show, learn=learn)
    all_peaks = np.array([initial_params])
    previous_regions = fl.regions_from_parameters(data_files[0].f, initial_params)
    if depth is None:
        depth = len(data_files)
    else:
        depth = min(depth, len(data_files))
    for i in util.progressbar(range(1, depth), 'Processing: '):
        sg.one_line_progress_meter('Overall Fitting Progress', i, depth, '-key-')
        f = data_files[i].f
        v = data_files[i].r
        reference_params = get_reference_params(all_peaks)
        noise_level = 3 * sw.extract_noise(v)
        active_params = fl.parameters_from_regions(f, v, previous_regions, noise_filter=noise_level)
        active_params = match_params(all_peaks[-1], active_params)
        if correction:
            active_params = find_missing_peaks(f, v, active_params, reference_params)
        all_peaks = np.append(all_peaks, np.array([active_params]), axis=0)
        active_params = util.remove_nans(active_params)
        previous_regions = fl.regions_from_parameters(data_files[i].f, active_params)
    sg.one_line_progress_meter_cancel('-key-')
    return all_peaks

def match_nearest(details, val):
    available_details = [[], []]
    for i in range(0, len(details[0])):
        if not details[2][i]:
            available_details[0].append(details[0][i])
            available_details[1].append(details[1][i])
    for i in range(0, len(available_details[0])):
        available_details[0][i] = np.abs(available_details[0][i] - val)
    val_min = min(available_details[0])
    ind_min = available_details[0].index(val_min)
    return available_details[1][ind_min]

def match_params(reference_params, new_params):
    if len(new_params) > len(reference_params):
        reference_params, new_params = new_params, reference_params
    ref_f0 = reference_params[:,1]
    new_f0 = new_params[:,1]
    f0_details = [[], [], []]
    for i in range(0, len(ref_f0)):
        f0_details[0].append(ref_f0[i])
        f0_details[1].append(i)
        if np.isnan(ref_f0[i]):
            f0_details[2].append(True)
        else:
            f0_details[2].append(False)
    if not all(np.isnan(new_f0)):
        for i in range(0, len(new_f0)):
            ind_min = match_nearest(f0_details, new_f0[i])
            f0_details[2][ind_min] = i
    return_params = np.empty((0, 4))
    for i in range(0, len(ref_f0)):
        if type(f0_details[2][i]) is int:
            p = new_params[f0_details[2][i]]
        else:
            p = np.array([np.nan, np.nan, np.nan, np.nan])
        return_params = np.append(return_params, np.array([p]), axis=0)
    return return_params

def update_params(param_list):
    for i in util.progressbar(range(1, len(param_list)), "Tracking: "):
        param_list[i] = match_params(param_list[i - 1], param_list[i])
    return param_list
