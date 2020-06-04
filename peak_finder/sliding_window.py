import numpy as np
from . import classify_data as cd
from . import utilities as util

# Break down into several different size scales
# For each size scale, slide over window and say if there is a Lorentzian there or not

def flip_bool(a):
    """
    Flips the bit of the provided input.
    It's recommended to use bit_invert() in utilities instead.
    """
    return (a + 1 - 2) * -1

def find_regions(has_lorentz):
    """
    Input: An array of 1s and 0s that specify whether a Lorentzian is detected within the corresponding window.
    Output: A region array denoting the starting and ending indices of each region with Lorentzians in it.
    """
    regions = np.empty((0, 2))
    good_indices = np.where(has_lorentz == 1)
    if len(good_indices[0]) < 1:
        return regions
    mark_indices = np.zeros(has_lorentz.shape)
    start_index = int(good_indices[0][0])
    end_index = int(good_indices[0][-1])
    mark_indices[start_index] = 1
    mark_indices[end_index] = 1
    last_value = has_lorentz[start_index]
    for i in range(start_index + 1, end_index):
        if last_value != has_lorentz[i]:
            mark_indices[i] = 1
        last_value = has_lorentz[i]
    change_indices = np.where(mark_indices == 1)[0]
    j = 0
    while j < len(change_indices) - 1:
        regions = np.append(regions, np.array([[change_indices[j], change_indices[j+1]]]), axis=0)
        j += 2
    return regions

def decompose_by_scale(v, zoom, scale=(0,1,1024), overlap=1/4):
    """
    Given the provided zoom level, will decompose into an array of many small windows.
    """
    windows = np.empty((0,scale[2]))
    v = cd.normalize_1d(v, scale=scale)
    window_size = 1 / zoom
    num_windows = int(np.ceil(zoom / overlap))
    for i in range(0, num_windows):
        if i / num_windows + window_size <= 1:
            window = cd.scale_zoom(v, i / num_windows, i / num_windows + window_size)
            window = cd.normalize_1d(window, scale=scale)
            windows = np.append(windows, np.array([window]), axis=0)
    return windows

def decompose_all(v, num_zooms, scale=(0,1,1024), overlap=1/4, zoom_level=2):
    """
    Will give a list of window arrays for all scales with the allowed zoom levels.
    """
    if num_zooms < 1:
        return [v]
    windows_list = []
    for i in range(0, num_zooms):
        windows_list.append(decompose_by_scale(v, zoom_level ** i, scale=scale, overlap=overlap))
    return windows_list

def compose_by_scale(labels, zoom, scale=(0,1,1024), overlap=1/4):
    """
    Given an array of windows at a specific zoom level, will return the corresponding region array.
    """
    window_size = np.ceil(scale[2] / zoom)
    offset = window_size * overlap
    has_lorentz = np.ones(scale[2])
    labels = flip_bool(labels)
    for i in range(0, len(labels)):
        start_index = int(i * offset)
        end_index = int(start_index + window_size)
        set_value = int(labels[i])
        set_indices = np.ones((1, end_index - start_index)) * set_value
        has_lorentz[start_index:end_index] = has_lorentz[start_index:end_index] * set_indices
    has_lorentz = flip_bool(has_lorentz)
    return find_regions(has_lorentz)

def compose_all(labels_list, scale=(0,1,1024), overlap=1/4, zoom_level=2):
    """
    Given a list of window arrays, will return the entire corresponding region array.
    """
    regions = np.empty((0, 2))
    for i in range(0, len(labels_list)):
        if labels_list[i] is not None:
            regions = np.append(regions, compose_by_scale(labels_list[i], zoom_level ** i, scale=scale, overlap=overlap), axis=0)
    return regions

def slide_scale(model, v, max_zoom=5, min_zoom=0, scale=(0,1,1024), overlap=1/4, zoom_level=2, confidence_tolerance=0.9, merge_tolerance=None, compress=True, target=1, fix_zoom=False):
    """
    Uses the specified model, displacement, and parameters to return all the detected Lorentzian regions.
    """
    if fix_zoom:
        max_zoom = min_zoom
    min_zoom -= 1
    final_scale = cd.scale_1d(v)
    if merge_tolerance is None:
        merge_tolerance = 0.8 * (1 - overlap)
    window_list = decompose_all(v, max_zoom, scale, overlap, zoom_level=zoom_level)
    prediction_list = []
    for i in range(0, len(window_list)):
        prediction_scores = model.predict(window_list[i])
        if target == 0:
            prediction_scores = util.bit_invert(prediction_scores)
        prediction_scores[:, 1] -= confidence_tolerance
        predictions = np.argmax(prediction_scores, axis=1)
        if i >= min_zoom:
            prediction_list.append(predictions)
        else:
            prediction_list.append(None)
    regions = compose_all(prediction_list, scale=scale, overlap=overlap, zoom_level=zoom_level)
    regions.sort(axis=0)
    if compress:
        regions = compress_regions(regions, merge_tolerance=merge_tolerance)
    regions = cd.normalize_index(regions, scale, final_scale)
    return regions

def check_overlap(r1, r2, merge_tolerance=0.6):
    """
    Determines if two Lorentzian regions overlap enough to be merged or not.
    """
    if r1[1] < r2[0]:
        return False
    else:
        r1_size = r1[1] - r1[0]
        r2_size = r2[1] - r2[0]
        overlap_min = r2[0]
        overlap_max = r1[1]
        covered = overlap_max - overlap_min
        r1_overlap = covered / r1_size
        r2_overlap = covered / r2_size
        cover_amount = max(r1_overlap, r2_overlap)
        return cover_amount > merge_tolerance

def compress_regions(regions, merge_tolerance=0.6):
    """
    Given a region array, will merge the ones that sufficiently overlap.
    """
    region_sets = []
    region_parent = []
    region_count = len(regions)
    for i in range(0, region_count):
        region_sets.append({(regions[i][0], regions[i][1])})
        region_parent.append(i)
    for i in range(0, region_count):
        for j in range(i, region_count):
            r1 = regions[i]
            r2 = regions[j]
            if check_overlap(r1, r2, merge_tolerance=merge_tolerance):
                region_parent[j] = region_parent[i]
    for j in range(region_count - 1, -1, -1):
        i = region_parent[j]
        if i != j:
            region_sets[i] = region_sets[i] | region_sets[j]
            region_sets.pop(j)
    for i in range(0, len(region_sets)):
        s = sorted(list(region_sets[i]), key=lambda r: r[1])
        compressed_regions = np.empty((0, 2))
        for r in s:
            r = np.array([r])
            compressed_regions = np.append(compressed_regions, r, axis=0)
        region_sets[i] = compressed_regions
    region_count = len(region_sets)
    final_regions = np.empty((0, 2))
    for i in range(0, region_count):
        r1 = region_sets[i][0][0]
        r2 = region_sets[i][-1][1]
        full_region = np.array([[r1, r2]])
        final_regions = np.append(final_regions, full_region, axis=0)
    return final_regions

def quick_reduce(v, reduce_zoom):
    """
    Input: A numpy array of some data and an amount to zoom in.
    Output: A list of numpy arrays of that data broken down by the specified zoom amount.
    """
    num_outputs = int(1 / reduce_zoom)
    start = np.arange(0, num_outputs) * reduce_zoom
    end = np.arange(1, num_outputs + 1) * reduce_zoom
    v_list = []
    for i in range(0, num_outputs):
        v_list.append(cd.scale_zoom(v, start[i], end[i]))
    return v_list

def reduce_window_plot(model, f, v, reduce_zoom=0.05, max_zoom=7, min_zoom=0, overlap=1/4, zoom_level=2, confidence_tolerance=0.95, merge_tolerance=0.4, compress=True):
    """
    Returns the region lists and corresponding displacements of those regions after running a sliding window detection.
    """
    v_list = quick_reduce(v, reduce_zoom=reduce_zoom)
    regions_list = []
    for cut_v in v_list:
        cut_regions = slide_scale(model=model, v=cut_v, max_zoom=max_zoom, min_zoom=min_zoom, overlap=overlap, zoom_level=zoom_level)
        regions_list.append(cut_regions)
    for i in range(0, len(regions_list)):
        regions_list[i] = regions_list[i] + (i * 1024)
    return regions_list, v_list

def split_peaks(model, f, v, min_zoom=3, max_zoom=6, min_data_points=10, confidence_tolerance=0.95, single_zoom=False):
    """
    Given a specified model, will try and return a new set of regions with peaks separated apart.
    """
    regions_by_zoom_list = []
    for i in range(min_zoom, max_zoom + 1):
        regions_by_zoom_list.append(util.drop_region(slide_scale(model, v, min_zoom=i, max_zoom=i, confidence_tolerance=confidence_tolerance, fix_zoom=True, target=1, overlap=1/4, zoom_level=2), min_length=min_data_points))
    if single_zoom:
        selected_regions = max(regions_by_zoom_list, key=lambda r: r.shape[0])
    else:
        selected_regions = np.empty((0, 2))
        for r in regions_by_zoom_list:
            selected_regions = np.append(selected_regions, r, axis=0)
    return selected_regions
    
def split_all_peaks(model, f, v, regions, min_zoom=3, max_zoom=6, min_data_points=10, confidence_tolerance=0.95, single_zoom=False):
    """
    Given a specified model, will try and return a new set of regions with peaks separated apart for all the provided regions.
    """
    split_regions = np.empty((0, 2))
    for i in range(0, len(regions)):
        split_f, split_v = util.extract_region(i, regions, f, v)
        split_regions = np.append(split_regions, split_peaks(model, split_f, split_v, min_zoom=min_zoom, max_zoom=max_zoom, min_data_points=min_data_points, confidence_tolerance=confidence_tolerance, single_zoom=single_zoom) + regions[i][0], axis=0)
    return util.drop_region(split_regions, min_length=min_data_points)

def extract_noise(v, depth=8):
    """
    Estimates a noise level given some displacement data and a number of indives to search for.
    """
    num_partitions = len(v) // depth
    v_truncated = v[0:num_partitions * depth]
    v_shaped = v_truncated.reshape((num_partitions, depth))
    noise_deltas = []
    local_v = []
    for i in range(0, num_partitions):
        coefficients = np.polyfit(np.arange(len(v_shaped[i])), v_shaped[i], 1)
        v_local = v_shaped[i] - coefficients[1] - coefficients[0] * np.arange(len(v_shaped[i]))
        max_v = max(v_local)
        min_v = min(v_local)
        noise_deltas.append(max_v - min_v)
        local_v.append(v_local)
    noise_deltas = np.array(noise_deltas)
    noise_deltas = noise_deltas[np.logical_not(np.isnan(noise_deltas))]
    return np.median(noise_deltas)