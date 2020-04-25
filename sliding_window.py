import numpy as np
import classify_data as cd

# Break down into several different size scales
# For each size scale, slide over window and say if there is a Lorentzian there or not

def flip_bool(a):
    return (a + 1 - 2) * -1

def find_regions(has_lorentz):
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
    windows = np.empty((0,scale[2]))
    v = cd.normalize_1d(v, scale=scale)
    window_size = np.ceil(scale[2] / zoom)
    offset = window_size * overlap
    num_windows = int(np.ceil(zoom / overlap))
    for i in range(0, num_windows):
        start_index = int(i * offset)
        end_index = int(start_index + window_size)
        if end_index <= scale[2]:
            window = v[start_index:end_index]
            window = cd.normalize_1d(window, scale=scale)
            windows = np.append(windows, np.array([window]), axis=0)
    return windows

def decompose_all(v, num_zooms, scale=(0,1,1024), overlap=1/4, zoom_level=2):
    if num_zooms < 1:
        return [v]
    windows_list = []
    for i in range(0, num_zooms):
        windows_list.append(decompose_by_scale(v, zoom_level ** i, scale=scale, overlap=overlap))
    return windows_list

def compose_by_scale(labels, zoom, scale=(0,1,1024), overlap=1/4):
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
    regions = np.empty((0, 2))
    for i in range(0, len(labels_list)):
        if labels_list[i] is not None:
            regions = np.append(regions, compose_by_scale(labels_list[i], zoom_level ** i, scale=scale, overlap=overlap), axis=0)
    return regions

def slide_scale(model, v, num_zooms=5, min_zoom=0, scale=(0,1,1024), overlap=1/4, zoom_level=2, confidence_tolerance=0.9, merge_tolerance=None, compress=True):
    if merge_tolerance is None:
        merge_tolerance = 0.8 * (1 - overlap)
    window_list = decompose_all(v, num_zooms, scale, overlap, zoom_level=zoom_level)
    prediction_list = []
    for i in range(0, len(window_list)):
        prediction_scores = model.predict(window_list[i])
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
    return regions

def check_overlap(r1, r2, merge_tolerance=0.6):
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