import numpy as np
import scipy.interpolate as interp

def arr_to_tup(a):
    return tuple(a.reshape(1, -1)[0])

def find_single_fit_range(lorentz_params):
    f0 = lorentz_params[1]
    FWHM = lorentz_params[2]
    return (f0 - 4 * FWHM, f0 - 2 * FWHM, f0 + 2 * FWHM, f0 + 4 * FWHM)

def find_full_fit_range(lorentz_params_array):
    (f_low_stop_list, f_low_list, f_high_list, f_high_stop_list) = find_all_fit_ranges(lorentz_params_array)
    f_low_stop = min(f_low_stop_list)
    f_low = min(f_low_list)
    f_high = max(f_high_list)
    f_high_stop = max(f_high_stop_list)
    return (f_low_stop, f_low, f_high, f_high_stop)

def find_all_fit_ranges(lorentz_params_array):
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
    if lorentz_params_1 is None or lorentz_params_2 is None:
        return False
    [low_lorentz, high_lorentz] = sorted([lorentz_params_1, lorentz_params_2], key=lambda l: l[1])
    low_fit_range = find_single_fit_range(low_lorentz)
    high_fit_range = find_single_fit_range(high_lorentz)
    return low_fit_range[2] > high_fit_range[1]

def partition_lorentz_params_array(lorentz_params_array):
    # Returns list of Lorentz arrays
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

def partition_class_data(block, lorentz_arrays_list, scale=(0,1,1024)):
    fit_range_list = find_partitioned_fit_ranges(lorentz_arrays_list)
    lorentz_arrays_list = block[1]
    data_arrays_list = block[2]
    block_size = len(lorentz_arrays_list)
    pass

def partition_data_2d(data_array, fit_range_list, lorentz_array_2d, scale=(0,1,1024)):
    scale_len = scale[2]
    lorentz_count_array = np.empty((0, 1))
    v_array = np.empty((0, scale_len))
    (f, v) = separate_data(data_array)
    ref_scale = scale_1d(f)
    temp_scale = (0, 1, ref_scale[2])
    # print('temp scale:' + str(temp_scale))
    for i in range(0, len(fit_range_list)):
        fit_range = fit_range_list[i]
        f_low_range = fit_range[1] - fit_range[0]
        f_high_range = fit_range[2] - fit_range[1]
        f_min = np.random.random() * f_low_range + fit_range[0]
        f_max = np.random.random() * f_high_range + fit_range[2]
        v_norm = normalize_1d(v, temp_scale)
        # print(v_norm)
        # print(f)
        # print(fit_range)
        # print(f_min)
        # print(f_max)
        np.putmask(v_norm, f<f_min, v_norm * 0 - 1)
        np.putmask(v_norm, f>f_max, v_norm * 0 -1)
        v_clipped = v_norm[v_norm >= 0]
        f_clipped = f[f >= f_min]
        f_clipped = f_clipped[f_clipped <= f_max]
        # print(v_clipped)
        v_norm = normalize_1d(v_clipped, scale)
        v_array = np.append(v_array, np.array([v_norm]), axis=0)
        num_lorentz = count_lorentz(fit_range, lorentz_array_2d)
        lorentz_count_array = np.append(lorentz_count_array, np.array([[num_lorentz]]), axis=0)
    return (v_array, lorentz_count_array)

def count_lorentz(fit_range, lorentz_array_2d):
    counter = 0
    for i in range(0, lorentz_array_2d.shape[0]):
        f0 = lorentz_array_2d[i][1]
        if f0 > fit_range[1] and f0 < fit_range[2]:
            counter += 1
    return counter

def find_partitioned_fit_ranges(lorentz_params_list):
    fit_range_list = []
    for a in lorentz_params_list:
        fit_range_list.append(find_full_fit_range(a))
    return fit_range_list

def evaluate_fit_range(predicted, fit_range):
    test1 = (predicted[0] >= fit_range[0])
    test2 = (predicted[0] <= fit_range[1])
    test3 = (predicted[1] >= fit_range[2])
    test4 = (predicted[1] <= fit_range[3])
    return all([test1, test2, test3, test4])

def evaluate_all_fit_ranges(predicted, fit_range_list):
    tests = []
    for i in range(0, len(fit_range_list)):
        tests.append(evaluate_fit_range(predicted[i], fit_range_list[i]))
    return all(tests)

def disect_lorentz_params_array(lorentz_params_array):
    if filter_lorentz(lorentz_params_array):
        lorentz_params_list = partition_lorentz_params_array(lorentz_params_array)
        fit_range_list = find_partitioned_fit_ranges(lorentz_params_list)
        num_fit_ranges = len(fit_range_list)
        return (num_fit_ranges, fit_range_list)
    else:
        return (0, None)

def scale_1d(x):
    return (min(x), max(x), len(x))

def normalize_1d(x, scale=(0,1,1024)):
    new_min = scale[0]
    new_max = scale[1]
    new_len = scale[2]
    (min_x, max_x, old_size) = scale_1d(x)
    x_norm = (x - min_x) / (max_x - min_x)
    old_baseline = np.linspace(0, 1, old_size)
    new_baseline = np.linspace(0, 1, new_len)
    # print(old_baseline)
    # print(x_norm)
    x_interp = interp.interp1d(old_baseline, x_norm)
    x_resized = (x_interp(new_baseline) * (new_max - new_min)) + new_min
    return x_resized

def normalize_0d(x, old_scale, new_scale=(0,1,1024)):
    old_delta = old_scale[1] - old_scale[0]
    new_delta = new_scale[1] - new_scale[0]
    old_min = old_scale[0]
    new_min = new_scale[0]
    return (x - old_min) * (new_delta / old_delta) + new_min

def normalize_lorentz_1d(lorentz, old_f_scale, old_v_scale, new_f_scale=(0,1,1024), new_v_scale=(0,1,1024)):
    A0 = lorentz[0]
    f0 = lorentz[1]
    G0 = lorentz[2]
    A1 = normalize_0d(A0, old_v_scale, new_v_scale)
    f1 = normalize_0d(f0, old_f_scale, new_f_scale)
    G1 = normalize_0d(G0, old_f_scale, new_f_scale)
    return np.array([A1, f1, G1, lorentz[3]])

def normalize_lorentz_2d(lorentz, old_f_scale, old_v_scale, new_f_scale=(0,1,1024), new_v_scale=(0,1,1024)):
    lorentz_array = np.empty((0, 4))
    for i in range(0, lorentz.shape[0]):
        l0 = lorentz[i]
        l1 = normalize_lorentz_1d(l0, old_f_scale, old_v_scale, new_f_scale, new_v_scale)
        lorentz_array = np.append(lorentz_array, np.array([l1]), axis=0)
    return lorentz_array

def filter_lorentz(lorentz_array):
    if lorentz_array is None:
        return False
    else:
        return lorentz_array.shape[1] == 4

def separate_data(f_v_array):
    f = f_v_array[:,0]
    v = f_v_array[:,1]
    return (f, v)

def merge_data(f, v):
    return np.transpose(np.vstack([f, v]))

def separate_all_data(data_arrays_list):
    separated_data_list = []
    for i in range(0, len(data_arrays_list)):
        separated_data_list.append(separate_data(data_arrays_list[i]))
    return separated_data_list

def pre_process_for_counting(block, scale=(0,1,1024)):
    lorentz_arrays_list = block[1]
    data_arrays_list = block[2]
    block_size = len(lorentz_arrays_list)
    scale_list = []
    processed_lorentz_arrays_list = []
    processed_data_array = np.empty((0, scale[2]))
    for i in range(0, block_size):
        lorentz_array = lorentz_arrays_list[i]
        data_array = data_arrays_list[i]
        (f_unprocessed, v_unprocessed) = separate_data(data_array)
        scale_f, scale_v = scale_1d(f_unprocessed), scale_1d(v_unprocessed)
        scale_list.append((scale_f, scale_v))
        f_processed = normalize_1d(f_unprocessed, scale)
        v_processed = normalize_1d(v_unprocessed, scale)
        if filter_lorentz(lorentz_array):
            l_processed = normalize_lorentz_2d(lorentz_array, scale_f, scale_v, scale, scale)
        else:
            l_processed = None
        processed_lorentz_arrays_list.append(l_processed)
        processed_data_array = np.append(processed_data_array, np.array([v_processed]), axis=0)
    results = (processed_lorentz_arrays_list, processed_data_array, scale_list)
    count_labels = []
    for i in range(0, len(processed_lorentz_arrays_list)):
        labels = disect_lorentz_params_array(processed_lorentz_arrays_list[i])
        count_labels.append(labels[0])
    count_labels = np.transpose(np.array([count_labels]))
    count_data = results[1]
    return count_labels, count_data

def pre_process_for_classifying(block, scale=(0,1,1024)):
    lorentz_arrays_list = block[1]
    data_arrays_list = block[2]
    block_size = len(lorentz_arrays_list)
    cluster_labels = np.empty((0, 1))
    cluster_data = np.empty((0, scale[2]))
    for i in range(0, block_size):
        lorentz_params = lorentz_arrays_list[i]
        f_v_data = data_arrays_list[i]
        fit_range_list = disect_lorentz_params_array(lorentz_params)[1]
        (v, labels) = partition_data_2d(f_v_data, fit_range_list, lorentz_params)
        cluster_labels = np.append(cluster_labels, labels, axis=0)
        cluster_data = np.append(cluster_data, v, axis=0)
    return cluster_labels, cluster_data