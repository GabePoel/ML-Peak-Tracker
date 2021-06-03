"""
Handles most if not all of the data preprocessing.
"""

import numpy as np
import multiprocessing as mp
import scipy.interpolate as interp

from . import utilities as util

def arr_to_tup(a):
    """
    Turns an array into a tuple.

    Parameters
    ----------
    a : arr
        The array to be turned into a tuple.

    Returns
    -------
    tuple
        The tuple form of the converted array.
    """
    return tuple(a.reshape(1, -1)[0])


def find_single_fit_range(lorentz_params):
    """
    Finds the fit range for a single Lorentzian.

    Parameters
    ----------
    lorentz_params : arr
        A 1D lorentzian parameter array.

    Returns
    -------
    tuple
        A two item tuple with the 0 index value being the start of the range
        and the 1 index being the end.
    """
    f0 = lorentz_params[1]
    FWHM = lorentz_params[2]
    return (f0 - 4 * FWHM, f0 - 2 * FWHM, f0 + 2 * FWHM, f0 + 4 * FWHM)


def find_full_fit_range(lorentz_params_array):
    """
    Finds a single fit range given an array defining a group of Lorentzians.

    Parameters
    ----------
    lorentz_params_array : arr
        A 2D Lorentzian parameter array.

    Returns
    -------
    tuple
        A combined fit range tuple.
    """
    (f_low_stop_list, f_low_list, f_high_list,
     f_high_stop_list) = find_all_fit_ranges(lorentz_params_array)
    f_low_stop = min(f_low_stop_list)
    f_low = min(f_low_list)
    f_high = max(f_high_list)
    f_high_stop = max(f_high_stop_list)
    return (f_low_stop, f_low, f_high, f_high_stop)


def find_all_fit_ranges(lorentz_params_array):
    """
    Returns a list of fit ranges for any number of Lorentzians.

    Parameters
    ----------
    lorentz_params_array : arr
        A 2D Lorentzian parameter array.

    Returns
    -------
    tuple
        A four element tuple that's analagous to a fit range. But each item
        is a list containing the corresponding item in a fit range for a given
        Lorentzian.
    """
    f_low_stop_list = []
    f_low_list = []
    f_high_list = []
    f_high_stop_list = []
    for i in range(0, lorentz_params_array.shape[0]):
        fit_range = find_single_fit_range(lorentz_params_array[i])
        f_low_stop_list.append(fit_range[0])
        f_low_list.append(fit_range[1])
        f_high_list.append(fit_range[2])
        f_high_stop_list.append(fit_range[3])
    return (f_low_stop_list, f_low_list, f_high_list, f_high_stop_list)


def check_overlap(lorentz_params_1, lorentz_params_2):
    """
    Checks if two Lorentzians overlap.

    Parameters
    ----------
    lorentz_params_1 : arr
    lorentz_params_2 : arr

    Returns
    -------
    bool
        True if the two Lorentzians sufficiently overlap. False if they don't.
    """
    if lorentz_params_1 is None or lorentz_params_2 is None:
        return False
    [low_lorentz, high_lorentz] = sorted(
        [lorentz_params_1, lorentz_params_2], key=lambda l: l[1])
    low_fit_range = find_single_fit_range(low_lorentz)
    high_fit_range = find_single_fit_range(high_lorentz)
    return low_fit_range[2] > high_fit_range[1]


def partition_lorentz_params_array(lorentz_params_array):
    """
    Returns list of Lorentz arrays.

    Parameters
    ----------
    lorentz_params_array : arr

    Returns
    -------
    list
        Each element in the list is a set of Lorentzians that are clustered
        together.
    """
    lorentz_sets = []
    lorentz_parent = []
    for i in range(0, lorentz_params_array.shape[0]):
        lorentz_sets.append({arr_to_tup(lorentz_params_array[i])})
        lorentz_parent.append(i)
    lorentz_count = lorentz_params_array.shape[0]
    for i in range(0, lorentz_count):
        for j in range(i, lorentz_count):
            l1 = lorentz_params_array[i]
            l2 = lorentz_params_array[j]
            if check_overlap(l1, l2):
                lorentz_parent[j] = lorentz_parent[i]
    for j in range(lorentz_count - 1, -1, -1):
        i = lorentz_parent[j]
        if i != j:
            lorentz_sets[i] = lorentz_sets[i] | lorentz_sets[j]
            lorentz_sets.pop(j)
    for i in range(0, len(lorentz_sets)):
        s = sorted(list(lorentz_sets[i]), key=lambda l: l[1])
        lorentz_params = np.empty((0, 4))
        for l in s:
            l = np.array([l])
            lorentz_params = np.append(lorentz_params, l, axis=0)
        lorentz_sets[i] = lorentz_params
    lorentz_sets = sorted(lorentz_sets, key=lambda a: a[0][1])
    return lorentz_sets


def partition_data_2d(
    data_array,
    fit_range_list,
    lorentz_array_2d,
    scale=(
        0,
        1,
        1024)):
    """
    Partitions a complete generated data set and gives a tuple which has all
    the corresponding data for each Lorentzian cluster as well as how many
    Lorentzians are in each of these clusters.

    Parameters
    ----------
    data_array : arr
        An unedited array of generated data.
    fit_range_list : list
        A tuple of lists corresponding to parts of fit ranges. See
        find_all_fit_ranges for more details.
    lorentz_array_2d : arr
        A 2D Lorentzian array.
    scale : tuple, optional
        The scale to normalize the output data to.

    Returns
    -------
        tuple
            Element 0 is a 2D array with all the data for each Lorentzian
            cluster normalized by the scale factor. Element 1 is the number
            of Lorentzians in each cluster of the same row.
    """
    scale_len = scale[2]
    lorentz_count_array = np.empty((0, 1))
    v_array = np.empty((0, scale_len))
    (f, v) = separate_data(data_array)
    ref_scale = scale_1d(f)
    temp_scale = (0, 1, ref_scale[2])
    for i in range(0, len(fit_range_list)):
        fit_range = fit_range_list[i]
        f_low_range = fit_range[1] - fit_range[0]
        f_high_range = fit_range[2] - fit_range[1]
        f_min = np.random.random() * f_low_range + fit_range[0]
        f_max = np.random.random() * f_high_range + fit_range[2]
        v_norm = normalize_1d(v, temp_scale)
        np.putmask(v_norm, f < f_min, v_norm * 0 - 1)
        np.putmask(v_norm, f > f_max, v_norm * 0 - 1)
        v_clipped = v_norm[v_norm >= 0]
        f_clipped = f[f >= f_min]
        f_clipped = f_clipped[f_clipped <= f_max]
        v_norm = normalize_1d(v_clipped, scale)
        v_array = np.append(v_array, np.array([v_norm]), axis=0)
        num_lorentz = count_lorentz(fit_range, lorentz_array_2d)
        lorentz_count_array = np.append(
            lorentz_count_array, np.array([[num_lorentz]]), axis=0)
    return (v_array, lorentz_count_array)


def count_lorentz(fit_range, lorentz_array_2d):
    """
    Counts how many Lorentzians from the 2D array are within the fit range.

    Parameters
    ----------
    fit_range : tuple
        A fit range tuple.
    lorentz_array_2d : arr
        A 2D Lorentzian array.

    Returns
    -------
    int
        How many of the Lorentzians in the 2D array land within the fit range.
    """
    counter = 0
    for i in range(0, lorentz_array_2d.shape[0]):
        f0 = lorentz_array_2d[i][1]
        if f0 > fit_range[1] and f0 < fit_range[2]:
            counter += 1
    return counter


def find_partitioned_fit_ranges(lorentz_params_list):
    """
    Returns a list of fit ranges from the provided Lorentzians.

    Parameters
    ----------
    lorentz_params_list : list

    Returns
    -------
    list
        A list of all the fit ranges after partitioning the input Lorentzians.
    """
    fit_range_list = []
    for a in lorentz_params_list:
        fit_range_list.append(find_full_fit_range(a))
    return fit_range_list


def evaluate_fit_range(predicted, fit_range):
    """
    Determines if the provided fit ranges are violated.
    """
    test1 = (predicted[0] >= fit_range[0])
    test2 = (predicted[0] <= fit_range[1])
    test3 = (predicted[1] >= fit_range[2])
    test4 = (predicted[1] <= fit_range[3])
    return all([test1, test2, test3, test4])


def evaluate_all_fit_ranges(predicted, fit_range_list):
    """
    Given a list of fit ranges, will check all of them to see how much they
    overlap and are violated.
    """
    tests = []
    for i in range(0, len(fit_range_list)):
        tests.append(evaluate_fit_range(predicted[i], fit_range_list[i]))
    return all(tests)


def disect_lorentz_params_array(lorentz_params_array):
    """
    Returns the number of fit ranges and a list of them from the given array of
    Lorentzian parameters.
    """
    if filter_lorentz(lorentz_params_array):
        lorentz_params_list = partition_lorentz_params_array(
            lorentz_params_array)
        fit_range_list = find_partitioned_fit_ranges(lorentz_params_list)
        num_fit_ranges = len(fit_range_list)
        return (num_fit_ranges, fit_range_list)
    else:
        return (0, None)


def scale_1d(x):
    """
    Determines the scale values for the given 1D array of data.
    Scales are structured as (min value, max value, total number of indices).
    """
    return (min(x), max(x), len(x))


def normalize_index(x, input_scale, output_scale):
    """
    Normalizes indices around the provided scale values.
    """
    return np.round(x / input_scale[2] * output_scale[2])


def normalize_data(
        background_params,
        lorentz_params,
        f,
        v,
        scale=(0, 1, 1024)):
    """
    Normalizes a data set.
    """
    old_f_scale = scale_1d(f)
    old_v_scale = scale_1d(v)
    background_params = background_params[0]
    background_params_norm = normalize_lorentz_1d(
        background_params, old_f_scale, old_v_scale, scale, scale)
    lorentz_params_norm = normalize_lorentz_2d(
        lorentz_params, old_f_scale, old_v_scale, scale, scale)
    f_norm = normalize_1d(f, scale)
    v_norm = normalize_1d(v, scale)
    return np.array([background_params_norm]
                    ), lorentz_params_norm, f_norm, v_norm


def normalize_1d(x, scale=(0, 1, 1024)):
    """
    Normalizes a given array of data around the provided scale factor.
    """
    new_min = scale[0]
    new_max = scale[1]
    new_len = scale[2]
    (min_x, max_x, old_size) = scale_1d(x)
    x_norm = (x - min_x) / (max_x - min_x)
    old_baseline = np.linspace(0, 1, old_size)
    new_baseline = np.linspace(0, 1, new_len)
    if len(old_baseline) <= 1:
        old_baseline = np.array([0, 1])
        x_norm = np.array([1, 0])
    x_interp = interp.interp1d(old_baseline, x_norm)
    x_resized = (x_interp(new_baseline) * (new_max - new_min)) + new_min
    return x_resized


def normalize_0d(x, old_scale=(0, 1, 1024), new_scale=(0, 1, 1024)):
    """
    Normalizes a single value around the provided scale factor.
    """
    old_delta = old_scale[1] - old_scale[0]
    new_delta = new_scale[1] - new_scale[0]
    old_min = old_scale[0]
    new_min = new_scale[0]
    return (x - old_min) * (new_delta / old_delta) + new_min


def normalize_lorentz_1d(
    lorentz, old_f_scale, old_v_scale, new_f_scale=(
        0, 1, 1024), new_v_scale=(
            0, 1, 1024)):
    """
    Normalizes single Lorentzian parameters around the provided scale factors.
    """
    A0 = lorentz[0]
    f0 = lorentz[1]
    G0 = lorentz[2]
    A1 = normalize_0d(A0 + old_v_scale[0], old_v_scale, new_v_scale)
    f1 = normalize_0d(f0, old_f_scale, new_f_scale)
    G1 = normalize_0d(G0 + old_f_scale[0], old_f_scale, new_f_scale)
    return np.array([A1, f1, G1, lorentz[3]])


def normalize_lorentz_2d(
    lorentz, old_f_scale, old_v_scale, new_f_scale=(
        0, 1, 1024), new_v_scale=(
            0, 1, 1024)):
    """
    Normalizes an array of many Lorentzian parameters around the provided scale
    factors.
    """
    lorentz_array = np.empty((0, 4))
    for i in range(0, lorentz.shape[0]):
        l0 = lorentz[i]
        l1 = normalize_lorentz_1d(
            l0, old_f_scale, old_v_scale, new_f_scale, new_v_scale)
        lorentz_array = np.append(lorentz_array, np.array([l1]), axis=0)
    return lorentz_array


def filter_lorentz(lorentz_array):
    """
    Gets rid of empty Lorentzian parameter arrays.
    """
    if lorentz_array is None:
        return False
    else:
        return lorentz_array.shape[1] == 4


def separate_data(f_v_array):
    """
    Given a 2D array of frequency and displacement data, will split it down the
    two like a zipper.
    'ziiiiiiiiiipppppppppppp'
    """
    f = f_v_array[:, 0]
    v = f_v_array[:, 1]
    return (f, v)


def merge_data(f, v):
    """
    Given separate frequency and displacement data, will merge them together
    like a zipper.
    'ziiiiiiiiiipppppppppppp'
    """
    return np.transpose(np.vstack([f, v]))


def separate_all_data(data_arrays_list):
    """
    Separates all data from the provided list of data arrays.
    """
    separated_data_list = []
    for i in range(0, len(data_arrays_list)):
        separated_data_list.append(separate_data(data_arrays_list[i]))
    return separated_data_list


def equalize_data(class_labels, class_data):
    """
    Equalizes the provided data with the provided labels.
    """
    a1 = class_data[np.where(class_labels == 1)[0]]
    a2 = class_data[np.where(class_labels == 2)[0]]
    # a3 = class_data[np.where(class_labels == 3)[0]]
    # a4 = class_data[np.where(class_labels == 4)[0]]
    max_len = min(len(a1), len(a2))
    # max_len = min(len(a1), len(a2), len(a3), len(a4))
    a1 = a1[0:max_len]
    a2 = a2[0:max_len]
    # a3 = a3[0:max_len]
    # a4 = a4[0:max_len]
    arr = np.concatenate((a1, a2))
    # arr = np.concatenate((a1, a2, a3, a4))
    b1 = np.ones((max_len, 1)) * 0
    b2 = np.ones((max_len, 1)) * 1
    # b3 = np.ones((max_len, 1)) * 2
    # b4 = np.ones((max_len, 1)) * 3
    brr = np.concatenate((b1, b2))
    # brr = np.concatenate((b1, b2, b3, b4))
    combined_arr = np.append(brr, arr, axis=1)
    combined_arr = np.random.permutation(combined_arr)
    labels = np.transpose(combined_arr)[0]
    data = np.delete(combined_arr, 0, axis=1)
    return labels, data


def pre_process_for_counting(block, scale=(0, 1, 1024)):
    """
    Pre-processes a data set so that it's ready for a counting model to be run
    or trained on it.
    """
    lorentz_arrays_list = block[1]
    data_arrays_list = block[2]
    block_size = len(lorentz_arrays_list)
    scale_list = []
    processed_lorentz_arrays_list = []
    processed_data_array = np.empty((0, scale[2]))
    for i in util._progressbar(range(block_size), "Normalizing: ", 40):
        lorentz_array = lorentz_arrays_list[i]
        data_array = data_arrays_list[i]
        (f_unprocessed, v_unprocessed) = separate_data(data_array)
        scale_f, scale_v = scale_1d(f_unprocessed), scale_1d(v_unprocessed)
        scale_list.append((scale_f, scale_v))
        f_processed = normalize_1d(f_unprocessed, scale)
        v_processed = normalize_1d(v_unprocessed, scale)
        if filter_lorentz(lorentz_array):
            l_processed = normalize_lorentz_2d(
                lorentz_array, scale_f, scale_v, scale, scale)
        else:
            l_processed = None
        processed_lorentz_arrays_list.append(l_processed)
        processed_data_array = np.append(
            processed_data_array, np.array([v_processed]), axis=0)
    results = (processed_lorentz_arrays_list, processed_data_array, scale_list)
    count_labels = []
    pro_length = len(processed_lorentz_arrays_list)
    for i in util._progressbar(range(pro_length), "Labeling: ", 40):
        labels = disect_lorentz_params_array(processed_lorentz_arrays_list[i])
        count_labels.append(labels[0])
    count_labels = np.transpose(np.array([count_labels]))
    count_data = results[1]
    return count_labels, count_data


def classify(input):
    i = input[0]
    lorentz_arrays_list = input[1]
    data_arrays_list = input[2]
    lorentz_params = lorentz_arrays_list[i]
    f_v_data = data_arrays_list[i]
    fit_range_list = disect_lorentz_params_array(lorentz_params)[1]
    (v, labels) = partition_data_2d(f_v_data, fit_range_list, lorentz_params)
    return (labels, v)


def pre_process_for_classifying(block, scale=(0, 1, 1024)):
    """
    Pre-processes a data set so that it's ready for a classifying model to be
    run or trained on it.
    """
    print('\nClassifying Data')
    lorentz_arrays_list = block[1]
    data_arrays_list = block[2]
    block_size = len(lorentz_arrays_list)
    cluster_labels = np.empty((0, 1))
    cluster_data = np.empty((0, scale[2]))
    pool = mp.Pool(mp.cpu_count())
    # for i in util._progressbar(range(block_size), "Classifying: ", 40):
    #     lorentz_params = lorentz_arrays_list[i]
    #     f_v_data = data_arrays_list[i]
    #     fit_range_list = disect_lorentz_params_array(lorentz_params)[1]
    #     (v, labels) = partition_data_2d(f_v_data, fit_range_list, lorentz_params)
    #     cluster_labels = np.append(cluster_labels, labels, axis=0)
    #     cluster_data = np.append(cluster_data, v, axis=0)
    results = pool.map(
        classify, [
            (i, lorentz_arrays_list, data_arrays_list) for i in range(block_size)])
    pool.close()
    for result in results:
        cluster_labels = np.append(cluster_labels, result[0], axis=0)
        cluster_data = np.append(cluster_data, result[1], axis=0)
    return cluster_labels, cluster_data


def pre_process_for_equal_classifying(block, scale=(0, 1, 1024)):
    """
    Pre-processes a data set so that it's ready for an equal classifying model
    to be run or trained on it.
    """
    class_labels, class_data = pre_process_for_classifying(block, scale)
    eq_labels, eq_data = equalize_data(class_labels, class_data)
    return eq_labels, eq_data


def scale_zoom(x, start, end):
    """
    Zooms in on the data from the provided array given the start and end values
    (not indices).
    """
    length = len(x)
    start_index = int(np.round(length * start))
    end_index = int(np.round(length * end))
    if start_index >= end_index:
        if start_index <= 3:
            start_index = 0
            end_index = 3
        else:
            start_index = end_index - 3
    return normalize_1d(x[start_index:end_index])
