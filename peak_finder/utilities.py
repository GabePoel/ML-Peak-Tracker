import sys
import os
import pickle
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from nptdms import TdmsFile

# Holds utilities that many parts of the peak tracker use.

class Data_File:
    """
    Stores the x, y, f, and v data for an imported tdms file.
    Here v is defined as v = sqrt(x ** 2 + y ** 2).
    """
    def __init__(self, x, y, f):
        self.x = x
        self.y = y
        self.f = f
        self.r = np.sqrt(x ** 2 + y ** 2)

    def import_probe_temp(self, probe_temp):
        self.probe_temp = probe_temp

    def import_cryo_temp(self, cryo_temp):
        self.cryo_temp = cryo_temp

    def import_meta(self, stamp):
        try:
            self.date, self.time, self.start_temp, self.end_temp = stamp.split('_')
        except:
            self.date, self.time, self.start_temp = stamp.split('_')
            self.end_temp = self.start_temp

def remove_nans(arr):
    if len(arr.shape) == 1:
        return arr[~np.isnan(arr)]
    elif len(arr.shape) == 2:
        new_arr = np.empty((0, len(arr[0])))
        for i in range(0, len(arr)):
            if not any(np.isnan(arr[i])):
                new_arr = np.append(new_arr, np.array([arr[i]]), axis=0)
        return new_arr

def progressbar(it, prefix="", size=60, file=sys.stdout, progress=True):
    """Use with an iteratore as 'it' to show a progress bar while waiting."""
    count = len(it)
    def show(j):
        try:
            x = int(size*j/count)
        except:
            x = 0
        if progress:
            print("%s[%s%s] %i/%i\r\r" % (prefix, "="*x, "-"*(size-x), j, count), end='\r')
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)

def load_file(path=None):
    if path is None:
        path = filedialog.askopenfilename()
    return path

def load_dir(path=None):
    if path is None:
        path = filedialog.askdirectory()
    return path

def import_tdms_file(path=None, show=False):
    """
    Just a wrapper for import_file. Calling it from its original name is depricated and will be removed.
    """
    return import_file(path=path, show=show)

def import_file(path=None, show=False):
    """
    This is deprecated and will be removed. Please call import_tdms_file() instead.
    Import a tdms file. Returns a Data_File object.
    Leave path blank to open a file dialog window and select the file manually. Otherwise pass in a path.
    """
    if path is None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename()
    tdms_file = TdmsFile.read(path)
    channels = tdms_file.groups()[0].channels()
    tdms_f = remove_nans(channels[0][:])
    tdms_x = remove_nans(channels[1][:])
    tdms_y = remove_nans(channels[2][:])
    tdms_file = Data_File(tdms_x, tdms_y, tdms_f)
    if len(channels) >= 5:
        tdms_probe_temp = remove_nans(channels[3][:])
        tdms_cryo_temp = remove_nans(channels[4][:])
        tdms_file.import_probe_temp(tdms_probe_temp)
        tdms_file.import_cryo_temp(tdms_cryo_temp)
    if show:
        print('Imported file from ' + str(path))
    return tdms_file

def plot_region(i, regions, f, v, color=None, show_boundaries=False, min_color='g', max_color='g'):
    """
    Given a region array, some frequency data, and some displacement data, will plot the specified index.
    """
    min_f = int(regions[i][0])
    max_f = int(regions[i][1])
    if color is None:
        plt.plot(f[min_f:max_f], v[min_f:max_f])
    else:
        plt.plot(f[min_f:max_f], v[min_f:max_f], color=color)
    if show_boundaries:
        plt.axvline(x=f[min_f], color=min_color)
        plt.axvline(x=f[max_f], color=max_color)

def extract_region(i, regions, f, v):
    """
    Given a region array, some frequency data, and some displacement data, will return the frequency and displacement for the region of the specified index.
    """
    min_f = int(regions[i][0])
    max_f = int(regions[i][1])
    region_f = f[min_f:max_f]
    region_v = v[min_f:max_f]
    return region_f, region_v

def bit_invert(b):
    """
    Inverts the provided bits.
    """
    return np.abs(b - 1)

def drop_region(regions, min_length=10, max_length=None):
    """
    Given a region array, will return only the regions with more than the min_length number of indices.
    """
    kept_regions = np.empty((0, 2))
    if max_length is None:
        max_length = np.inf
    for i in range(0, len(regions)):
        region = regions[i]
        if region[1] - region[0] >= min_length and region[1] - region[0] <= max_length:
            kept_regions = np.append(kept_regions, np.array([region]), axis=0)
    return kept_regions

def order_difference(val_1, val_2):
    """
    Returns how many orders of magnitude val_1 and val_2 differ by.
    """
    return np.abs(np.log10(val_1) - np.log10(val_2))

def compare_lorentz(l1, l2, f):
    """
    Input: Two Lorentzian parameter arrays ([A, f0, FWHM, phase]) and frequency data.
    Output: A numercial value of how similiar the Lorentzians are to each other.
    """
    f1 = l1[1]
    f2 = l2[1]
    delta_f0 = np.abs(f2 - f1)
    delta_f = max(f) - min(f)
    ind_per_f = len(f) / delta_f
    delta_ind = ind_per_f * delta_f0
    return np.abs(delta_ind)

def find_nearest_index(arr, val):
    """
    Input: An array and a value.
    Output: The index of the value in the array closest to that value.
    """
    reduced_arr = np.abs(arr - val)
    min_reduced_val = min(reduced_arr)
    return np.where(reduced_arr == min_reduced_val)[0][0]

def simplify_regions(regions):
    region_line = []
    for i in range(0, len(regions)):
        region_line.append([regions[i][0], 'start', False])
        region_line.append([regions[i][1], 'end', False])
    def take_first(elem):
        return elem[0]
    region_line.sort(key=take_first)
    j = 0
    last_index = None
    while j < len(region_line):
        if j == len(region_line) - 1:
            next_index = None
        else:
            next_index = region_line[j + 1][1]
        this_index = region_line[j][1]
        if last_index is None or next_index is None:
            region_line[j][2] = True
        elif this_index == 'start' and last_index == 'end':
            region_line[j][2] = True
        elif this_index == 'end' and next_index == 'start':
            region_line[j][2] = True
        last_index = this_index
        j += 1
    region_starts = []
    region_ends = []
    for i in range(0, len(region_line)):
        if region_line[i][2]:
            if region_line[i][1] == 'start':
                region_starts.append(region_line[i][0])
            else:
                region_ends.append(region_line[i][0])
    simplified_regions = np.empty((0, 2))
    for i in range(0, len(region_starts)):
        start = region_starts[i]
        end = region_ends[i]
        simplified_regions = np.append(simplified_regions, np.array([[start, end]]), axis=0)
    return simplified_regions

def param_sort(params_2d):
    not_nan_params = np.empty((0, 4))
    nan_params = np.empty((0, 4))
    nan_indices = []
    for i in range(0, len(params_2d)):
        if any(np.isnan(params_2d[i])):
            nan_indices.append(i)
            nan_params = np.append(nan_params, [params_2d[i]], axis=0)
        else:
            not_nan_params = np.append(not_nan_params, [params_2d[i]], axis=0)
    not_nan_params = not_nan_params[not_nan_params[:,1].argsort()]
    final_params = np.empty((0, 4))
    offset = 0
    for i in range(0, len(params_2d)):
        if i in nan_indices:
            offset += 1
            final_params = np.append(final_params, [[np.nan, np.nan, np.nan, np.nan]], axis=0)
        else:
            final_params = np.append(final_params, [not_nan_params[i - offset]], axis=0)
    return final_params

def append_params_3d(p1, p2):
    """
    Adds new Lorentzians to an existing set of parameters.
    Does not add later data points of existing Lorentzians!
    """
    p3 = []
    if not len(p1) == len(p2):
        raise ValueError("Parameters not of the same length.")
    else:
        for i in range(0, len(p1)):
            combined = np.append(p1[i], p2[i], axis=0)
            p3.append(combined)
    return np.array(p3)

def save(object, path=None):
    if path is None:
        path = filedialog.asksaveasfilename(filetypes = (("python objects", "*.pkl"), ("all files", "*.*")))
    pickle.dump(object, open(path, "wb"))
    return path

def load(path=None):
    if path is None:
        path = filedialog.askopenfilename()
    return pickle.load(open(path, "rb"))

def import_tdms_files(path=None, show=True):
    """
    Makes a list out of all the imported tdms files in chosen directory.
    """
    path = load_dir(path)
    names = os.listdir(path)
    data_files = []
    if show:
        print(path)
    for name in names:
        if name[-5:] == '.tdms':
            stamp = name[:-5]
            file_path = os.path.join(path, name)
            data_file = import_file(file_path)
            data_file.import_meta(stamp)
            data_files.append(data_file)
    data_files.sort(key=lambda d: int(str(d.date) + str(d.time)))
    return data_files

def get_temperatures(data_files):
    """
    Get a temperature array from a list of data files.
    """
    temperatures = []
    for i in range(0, len(data_files)):
        temperatures.append(float(data_files[i].start_temp[0:-1]))
    return np.array(temperatures)

def matplotlib_mac_fix():
    import matplotlib
    import importlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    importlib.reload(plt)