"""
Tools for tracking peaks in later sweeps based on peaks in earlier sweeps.
"""

import PySimpleGUI as sg
import numpy as np
from lorentzian_models import models

from . import automatic
from . import fit_lorentz as fl
from . import live_fitting as lf
from . import sliding_window as sw
from . import utilities as util


def find_last_peak(params_3d, index):
    """
    Finds the parameters for the last time a given Lorentzian wasn't a nan
    array.

    Parameters
    ----------
    params_3d : arr
        3D Lorentzian parameter array.
    index : int
        Index corresponding to the 2D Lorentzian parameter array to extract the
        last peak from.

    Returns
    -------
    arr
        1D Lorentzian parameter array for the last peak from the chosen index.
    arr
        Array with the number of recursion steps.
    """
    # print(params_3d)
    if not any(np.isnan(params_3d[-1][index])):
        # This is gross. It should be fixed at some point and made cleaner.
        return_params_1d = params_3d[-1][index]
        count_1d = np.array([1])
        if len(params_3d) > 1:
            comp_params, comp_count = find_last_peak(params_3d[0:-1], index)
            if (return_params_1d == comp_params).all():
                return_params_1d, count_1d = find_last_peak(
                    params_3d[0:-1], index)
    else:
        return_params_1d, count_1d = find_last_peak(params_3d[0:-1], index)
    return return_params_1d, count_1d + 1


def get_reference_params(params_3d):
    """
    Finds the last usable parameters for a complete 2d Lorentzian parameter
    array.

    Parameters
    ----------
    params_3d : arr
        3D Lorentzian parameter array.

    Returns
    -------
    arr
        2D Lorentzian parameter array.
    arr
        Step count.
    """
    reference_params_2d = np.empty((0, 4))
    step_count = np.empty((0, 1))
    for i in range(0, len(params_3d[0])):
        params_1d, count_1d = find_last_peak(params_3d, i)
        reference_params_2d = np.append(
            reference_params_2d, np.array([params_1d]), axis=0)
        step_count = np.append(step_count, [count_1d], axis=0)
    return reference_params_2d, step_count


def find_min_f(f, active_params, active_index):
    """
    Finds lowest f that doesn't intersect another Lorentzian.

    Parameters
    ----------
    f : arr
        1D array of frequencies.
    active_params : arr
        Lorentzian parameter array corresponding to the given frequency array.
    active_index : int
        Recursion index.

    Returns
    -------
    float
        Minimum frequency.
    """
    if active_index == 0:
        return min(f)
    elif not np.isnan(active_params[active_index - 1][1]):
        return active_params[active_index - 1][1] + \
            active_params[active_index - 1][2]
    else:
        return find_min_f(f, active_params, active_index - 1)


def find_max_f(f, active_params, active_index):
    """
    Finds greatest f that doesn't intersect another Lorentzian.

    Parameters
    ----------
    f : arr
        1D array of frequencies.
    active_params : arr
        Lorentzian parameter array corresponding to the given frequency array.
    active_index : int
        Recursion index.

    Returns
    -------
    float
        Minimum frequency.
    """
    if active_index == len(active_params) - 1:
        return max(f)
    elif not np.isnan(active_params[active_index + 1][1]):
        return active_params[active_index + 1][1] - \
            active_params[active_index + 1][2]
    else:
        return find_max_f(f, active_params, active_index + 1)


def find_extremes_f(f, active_params, active_index):
    """
    Finds values for an f region that doesn't intersect other Lorentzians.

    Parameters
    ----------
    f : arr
        1D array of frequencies.
    active_params : arr
        Lorentzian parameter array corresponding to the given frequency array.
    active_index : int
        Recursion index.

    Returns
    -------
    float
        Minimum frequency.
    """
    f1 = find_min_f(f, active_params, active_index)
    f2 = find_max_f(f, active_params, active_index)
    min_f = min(f1, f2)
    max_f = max(f1, f2)
    return min_f, max_f


def recursive_split(model, f, v, regions, base_zoom, limit=None):
    """
    Split regions recursively.

    Parameters
    ----------
    model : model
        Model to use.
    f : arr
        1D frequency array.
    v : arr
        1D amplitude (voltage) array.
    regions : arr
        Regions array.
    base_zoom : float
        Zoom level to recursively split over.
    limit : float
        The zoom limit level that ends the recursive splitting.

    Returns
    -------
    arr
        Recursively split regions.
    """
    if limit is None:
        limit = base_zoom - 2
    if base_zoom > 0:
        try:
            potential_regions = sw.split_peaks(
                model, f, v, regions, base_zoom - 1, base_zoom + 1)
        except BaseException:
            if base_zoom >= limit:
                potential_regions = recursive_split(
                    model, f, v, regions, base_zoom - 1, limit=limit)
            else:
                potential_regions = np.empty((0, 2))
    else:
        potential_regions = np.empty((0, 2))
    return potential_regions


def find_missing_peaks_sep(
        f,
        v,
        active_params,
        reference_params,
        search_limit):
    """
    Uses machine learning to search within a given frequency space for missing
    Lorentzians.

    Parameters
    ----------
    f : arr
        1D frequency array.
    v : arr
        1D amplitude (voltage) array.
    active_params : arr
        Lorentzian parameter array.
    reference_params : arr
        Lorentzian parameter array.
    search_limit : float
        Multiplicative factor saying how far out to search.
    """
    updated_params = np.empty((0, 4))
    noise_level = 3 * sw.extract_noise(f)
    model = models.tight_lorentzian()
    for i in range(0, len(active_params)):
        if not any(np.isnan(active_params[i])):
            updated_params = np.append(
                updated_params, np.array([active_params[i]]), axis=0)
        else:
            min_f, max_f = find_extremes_f(f, active_params, i)
            lim_f1 = reference_params[i][1] - \
                search_limit * reference_params[i][2]
            lim_f2 = reference_params[i][1] + \
                search_limit * reference_params[i][2]
            min_lim_f = min(lim_f1, lim_f2)
            max_lim_f = max(lim_f1, lim_f2)
            min_f = max(min_f, min_lim_f)
            max_f = min(max_f, max_lim_f)
            min_f, max_f = min(min_f, max_f), max(min_f, max_f)
            min_ind = util.find_nearest_index(f, min_f)
            max_ind = util.find_nearest_index(f, max_f)
            reference_FWHM = reference_params[i][2]
            delta_f = max_f - min_f
            try:
                base_zoom = np.abs(
                    int(np.round(np.log2(delta_f / reference_FWHM))))
            except BaseException:
                print(reference_params[i])
                print(delta_f)
                print(min_f)
                print(max_f)
                base_zoom = 8
            # make sure we don't waste computing resources looking at nothing
            if base_zoom > 8:
                base_zoom = 8
            if base_zoom < 2:
                base_zoom = 2
            regions = np.array([[min_ind, max_ind]])
            potential_regions = recursive_split(
                model, f, v, regions, base_zoom)
            if len(potential_regions) > 0:
                potential_params = fl.parameters_from_regions(
                    f, v, potential_regions, noise_filter=noise_level)
                if len(potential_params) > 0:
                    f0_vals = potential_params[:, 1]
                    reference_f0 = reference_params[i][1]
                    closest_index = util.find_nearest_index(
                        f0_vals, reference_f0)
                    closest_params = potential_params[closest_index]
                    updated_params = np.append(
                        updated_params, np.array([closest_params]), axis=0)
                else:
                    updated_params = np.append(updated_params, np.array(
                        [[np.nan, np.nan, np.nan, np.nan]]), axis=0)
            else:
                updated_params = np.append(updated_params, np.array(
                    [[np.nan, np.nan, np.nan, np.nan]]), axis=0)
    return updated_params


def find_missing_peaks_tog(
        f,
        v,
        active_params,
        reference_params,
        search_limit):
    if not any(np.isnan(active_params.flatten())):
        return active_params
    else:
        noise_level = 3 * sw.extract_noise(f)
        model = models.tight_lorentzian()
        nan_indices = []
        nan_limits = []
        regions = np.empty((0, 2))
        known_zooms = []
        for i in range(0, len(active_params)):
            if any(np.isnan(active_params[i])):
                nan_indices.append(i)
                min_f, max_f = find_extremes_f(f, active_params, i)
                lim_f1 = reference_params[i][1] - \
                    search_limit * reference_params[i][2]
                lim_f2 = reference_params[i][1] + \
                    search_limit * reference_params[i][2]
                min_lim_f = min(lim_f1, lim_f2)
                max_lim_f = max(lim_f1, lim_f2)
                min_f = max(min_f, min_lim_f)
                max_f = min(max_f, max_lim_f)
                min_f, max_f = min(min_f, max_f), max(min_f, max_f)
                nan_limits.append((min_f, max_f))
                min_ind = util.find_nearest_index(f, min_f)
                max_ind = util.find_nearest_index(f, max_f)
                reference_FWHM = reference_params[i][2]
                delta_f = max_f - min_f
                try:
                    base_zoom = np.abs(
                        int(np.ceil(np.log2(delta_f / reference_FWHM))))
                except BaseException:
                    base_zoom = 8
                if base_zoom > 8:
                    base_zoom = 8
                if base_zoom < 2:
                    base_zoom = 2
                known_zooms.append(base_zoom)
                known_zooms.append(base_zoom - 1)
                regions_one = np.array([[min_ind, max_ind]])
                regions = np.append(regions, regions_one, axis=0)
        input_regions = util.drop_region(regions)
        try:
            potential_regions = sw.split_peaks(
                model, f, v, input_regions, min(known_zooms), max(known_zooms))
        except BaseException:
            potential_regions = np.empty((0, 2))
        potential_params = fl.parameters_from_regions(
            f, v, potential_regions, noise_filter=noise_level)
        nan_params = np.empty((0, 4))
        for i in range(0, len(nan_indices)):
            new_params = np.array([reference_params[nan_indices[i]]])
            nan_params = np.append(nan_params, new_params, axis=0)
        chosen_params = match_params(nan_params, potential_params)
        checked_params = np.empty((0, 4))
        for i in range(0, len(nan_params)):
            if chosen_params[i][1] > nan_limits[i][0] and chosen_params[i][
                1] < nan_limits[i][1]:
                    checked_params = np.append(
                        checked_params, [chosen_params[i]], axis=0)
            else:
                checked_params = np.append(
                    checked_params, [[np.nan, np.nan, np.nan, np.nan]], axis=0)
        updated_params = nan_merge(active_params, checked_params)
        return updated_params


def extend_params(params_2d, final_length):
    nan_array = np.empty((0, 4))
    for i in range(0, final_length - len(params_2d)):
        nan_array = np.append(
            nan_array, [[np.nan, np.nan, np.nan, np.nan]], axis=0)
    return np.append(params_2d, nan_array, axis=0)


def track_temperatures_learn(data_files, depth=None, live_display=False):
    sg.one_line_progress_meter_cancel('-key-')
    live = None
    all_peaks = []
    if depth is None:
        depth = len(data_files)
    for i in util._progressbar(range(0, depth), 'Processing: '):
        sg.one_line_progress_meter(
            'Overall Fitting Progress', i, depth, '-key-')
        parameters = automatic.quick_analyze(
            data_files[i].f, data_files[i].r, show=False, learn=True)
        if live_display:
            if live is not None:
                live.close_window()
            live = lf.Live_Instance(data_files[i].f, data_files[i].r)
            live.import_lorentzians(parameters)
            live.activate(loop=False)
        all_peaks.append(parameters)
    most_peaks_param = max(all_peaks, key=lambda a: len(a))
    most_peaks_length = len(most_peaks_param)
    for i in range(0, len(all_peaks)):
        all_peaks[i] = extend_params(all_peaks[i], most_peaks_length)
    all_peaks = np.array(all_peaks)
    sg.one_line_progress_meter_cancel('-key-')
    return all_peaks


def track_temperatures(
        data_files,
        initial_params=None,
        show=True,
        depth=None,
        learn=True,
        correction=True,
        start_index=0,
        print_params=False,
        search_limit=None,
        noise_limit=3,
        method='together',
        live_display=False,
        progress_window=True):
    """
    Tracks Lorentzians over a complete temperature sweep from some optional
    initial parameters.
    """
    # previous refers to the index before
    # reference refers to the last known usable values (could be index before or
    # earlier)
    # active refers to the current index
    # search_limit doesn't currently do anything
    if progress_window:
        sg.one_line_progress_meter_cancel('-key-')
    live = None
    if initial_params is None:
        initial_params = np.empty((0, 4))
        initial_f = data_files[start_index].f
        initial_v = data_files[start_index].r
        initial_params = automatic.quick_analyze(
            initial_f, initial_v, show=show, learn=learn)
    if search_limit is None:
        search_limit = np.inf
    if method == 'separate':
        find_missing_peaks = find_missing_peaks_sep
    elif method == 'together':
        find_missing_peaks = find_missing_peaks_tog
    else:
        print('No valid peak refinding method given. Not doing any correction.')
        correction = False
    initial_params = util.param_sort(initial_params)
    all_peaks = np.array([initial_params])
    previous_regions = fl.regions_from_parameters(
        data_files[start_index].f, initial_params)
    if depth is None:
        depth = len(data_files) - start_index
    else:
        depth = min(depth, len(data_files) - start_index)
    if print_params:
        print('Initial Parameters: ')
        print(initial_params)
    for i in util._progressbar(
            range(start_index + 1, start_index + depth),
            'Processing: '):
        if progress_window:
            sg.one_line_progress_meter(
                'Overall Fitting Progress', i - start_index, depth, '-key-')
        f = data_files[i].f
        v = data_files[i].r
        reference_params, reference_counts = get_reference_params(all_peaks)
        noise_level = noise_limit * sw.extract_noise(v)
        active_params = fl.parameters_from_regions(
            f, v, previous_regions, noise_filter=noise_level)
        active_params = match_params(all_peaks[-1], active_params)
        active_params = util.param_sort(active_params)
        if correction:
            active_params = find_missing_peaks(
                f, v, active_params, reference_params, search_limit)
            active_params = match_params(reference_params, active_params)
            active_params = util.param_sort(active_params)
        active_params = util.param_sort(active_params)
        active_params = bind_params(
            reference_params, active_params, reference_counts)
        all_peaks = np.append(all_peaks, np.array([active_params]), axis=0)
        region_params, region_counts = get_reference_params(all_peaks)
        if live_display:
            if live is not None:
                live.close_window()
            live = lf.Live_Instance(f, v)
            live.import_lorentzians(region_params)
            live.activate(loop=False)
        if any(np.isnan(region_params.flatten())):
            region_params = util.remove_nans(region_params)
        previous_regions = fl.regions_from_parameters(
            data_files[i].f, region_params)
        previous_regions = bind_regions(f, previous_regions, region_params)
        previous_regions = util.drop_region(previous_regions)
    if live is not None:
        live.close_window()
    if progress_window:
        sg.one_line_progress_meter_cancel('-key-')
    return all_peaks


def bind_regions(f, regions, parameters):
    # regions and parameters must be the same length
    bound_regions = np.empty((0, 2))
    for i in range(0, len(regions)):
        if i == 0:
            min_bound = 0
        else:
            min_bound = util.find_nearest_index(f, parameters[i - 1][1])
        if i == len(regions) - 1:
            max_bound = len(regions)
        else:
            max_bound = util.find_nearest_index(f, parameters[i + 1][1])
        min_ind = regions[i][0]
        max_ind = regions[i][1]
        min_ind = max(min_ind, min_bound)
        max_ind = min(max_ind, max_bound)
        bound_regions = np.append(bound_regions, [[min_ind, max_ind]], axis=0)
    return bound_regions


def bind_params(reference_params_2d, target_params_2d, reference_counts):
    bound_params = np.empty((0, 4))
    for i in range(0, len(reference_params_2d)):
        if params_bound_check(
                reference_params_2d[i],
                target_params_2d[i],
                reference_counts[i]):
            bound_params = np.append(
                bound_params, [target_params_2d[i]], axis=0)
        else:
            bound_params = np.append(
                bound_params, [[np.nan, np.nan, np.nan, np.nan]], axis=0)
    return bound_params


def params_bound_check(
        reference_params_1d,
        target_params_1d,
        reference_counts_1d):
    check_1 = bound_A_check(reference_params_1d,
                            target_params_1d, reference_counts_1d)
    check_2 = bound_f0_check(
        reference_params_1d, target_params_1d, reference_counts_1d)
    check_3 = bound_FWHM_check(
        reference_params_1d, target_params_1d, reference_counts_1d)
    check_4 = bound_phase_check(
        reference_params_1d, target_params_1d, reference_counts_1d)
    checks = [check_1, check_2, check_3, check_4]
    return all(checks)


def bound_A_check(reference_params_1d, target_params_1d, reference_counts_1d):
    return util.order_difference(
        reference_params_1d[0], target_params_1d[0]) < .35 * (1 + 
            reference_counts_1d / 3)


def bound_f0_check(reference_params_1d, target_params_1d, reference_counts_1d):
    delta_f = np.abs(reference_params_1d[1] - target_params_1d[1])
    allowed_delta_f = np.abs(2 * reference_params_1d[2]) * reference_counts_1d
    return delta_f < allowed_delta_f


def bound_FWHM_check(
        refrence_params_1d,
        target_params_1d,
        reference_counts_1d):
    return util.order_difference(
        refrence_params_1d[2],
        target_params_1d[2]) < .5 * reference_counts_1d


def bound_phase_check(
        reference_params_1d,
        target_params_1d,
        reference_counts_1d):
    return np.abs(
        reference_params_1d[3] - target_params_1d[3]) < (reference_counts_1d *
            np.pi / 2)


def nan_merge(main_params, input_params):
    final_params = np.empty((0, 4))
    offset = 0
    for i in range(0, len(main_params)):
        if not any(np.isnan(main_params[i])):
            final_params = np.append(final_params, [main_params[i]], axis=0)
        else:
            final_params = np.append(
                final_params, [input_params[offset]], axis=0)
            offset += 1
    return final_params


def match_params(reference_params, target_params):
    """
    Matches the parameters between two parameter arrays.

    Parameters
    ----------
    reference_params : arr
        2D Lorentzian parameter array.
    target_params : arr
        2D Lorentzian parameter array.

    Returns
    -------
    updated_params : arr
        2D Lorentzian parameter array. Revised version of target_params.
    """
    ref_invert = False
    if len(target_params) > len(reference_params):
        reference_params, target_params = target_params, reference_params
        ref_invert = True
    reference_list = []
    target_list = []
    match_table = []
    # [ identifying index, parameters, matching index ]
    for i in range(0, len(reference_params)):
        reference_list.append([i, reference_params[i], None])
    for i in range(0, len(target_params)):
        target_list.append([i, target_params[i], None])
    for i in range(0, len(reference_params)):
        match_table.append([])
        for j in range(0, len(target_params)):
            ref_f0 = reference_params[i][1]
            tar_f0 = target_params[j][1]
            match_table[i].append(np.abs(ref_f0 - tar_f0))
    match_array = np.array(match_table)
    match_shape = match_array.shape
    best_vals = np.dstack(np.unravel_index(
        np.argsort(match_array.ravel()), match_shape))[0]
    num_matched = 0
    k_best = 0
    while num_matched < len(target_params):
        indices = best_vals[k_best]
        i = indices[0]
        j = indices[1]
        if update_match(reference_list, target_list, i, j):
            num_matched += 1
        k_best += 1
    if ref_invert:
        updated_params = get_updated_target(target_list, reference_list)
    else:
        updated_params = get_updated_target(reference_list, target_list)
    return updated_params


def update_match(reference_list, target_list, i, j):
    if reference_list[i][2] is None and target_list[j][2] is None:
        reference_list[i][2] = j
        target_list[j][2] = i
        return True
    else:
        return False


def get_updated_target(reference_list, target_list):
    updated_params = np.empty((0, 4))
    for i in range(0, len(reference_list)):
        j = reference_list[i][2]
        if j is not None:
            params_to_add = np.array([target_list[j][1]])
        else:
            params_to_add = np.array([[np.nan, np.nan, np.nan, np.nan]])
        updated_params = np.append(updated_params, params_to_add, axis=0)
    return updated_params


def update_params(param_list):
    """
    Given a list of parameters, will match them all up in order. See
    match_params for additional context.
    """
    for i in util._progressbar(range(1, len(param_list)), "Tracking: "):
        param_list[i] = match_params(param_list[i - 1], param_list[i])
    return param_list


def final_sorting(params_3d):
    """
    Sorts a 3D Lorentzian parameter array.

    Parameters
    ----------
    params_3d : arr
        3D Lorentzian parameter array.

    Returns
    -------
    final_params : arr
        3D Lorentzian parameter array.
    """
    final_params = np.array([params_3d[0]])
    for i in range(1, len(params_3d)):
        reference_params, counts = get_reference_params(params_3d[0:i])
        target_params = params_3d[i]
        matched_params = match_params(reference_params, target_params)
        combined_params = np.append(
            [reference_params], [target_params], axis=0)
        matched_params, counts = get_reference_params(combined_params)
        final_params = np.append(final_params, [matched_params], axis=0)
    return final_params
