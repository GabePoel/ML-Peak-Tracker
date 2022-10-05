__all__ = [
    '_bit_invert',
    '_matplotlib_mac_fix',
    '_progressbar',
    '_save_file',
    '_scatter_pts',
    '_simplify_regions',
    'append_params_3d',
    'attach_temps_to_parameters',
    'compare_lorentz',
    'Data_File',
    'delete_parameters_from_f_regions_2d',
    'delete_parameters_from_f_regions_3d',
    'delete_parameters',
    'drop_region',
    'extract_region',
    'find_missing_1d',
    'find_missing_2d',
    'find_missing_3d',
    'find_nearest_index',
    'freq_sort_2d',
    'get_all_params',
    'get_freqs',
    'get_temperatures',
    'get_temps_and_freqs',
    'import_tdms_dir',
    'import_tdms_file',
    'import_tdms_files',
    'install_models',
    'load_dir',
    'load_file',
    'load_files',
    'load_freqs_with_temps',
    'load_freqs',
    'load',
    'match_lengths',
    'normalize_1d',
    'order_difference',
    'param_sort',
    'plot_region',
    'remove_nans',
    'save_freqs_with_temps',
    'save_freqs',
    'save_Tf',
    'save',
    'scale_1d',
    'set_all_params',
    'temp_freq_sort_2d',
]

import sys
import os
import pickle
import tkinter as tk
# from typing import overload
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from tkinter import filedialog
from nptdms import TdmsFile
from time import sleep
from tqdm import tqdm


def _open_file():
    """
    Tkinter backend file dialog for opening a single file.
    """
    tk.Tk().withdraw()
    return filedialog.askopenfilename()


def _open_files():
    """
    Tkinter backend file dialog for opening multiple files.
    """
    tk.Tk().withdraw()
    return filedialog.askopenfilenames()


def _save_file(filters=[], name=''):
    """
    Tkinter backend file dialog for saving a single file.
    """
    tk.Tk().withdraw()
    return filedialog.asksaveasfilename(
        filetypes=filters, initialfile=name)


def _open_folder():
    """
    Tkinter backend file dialog for opening multiple files.
    """
    tk.Tk().withdraw()
    return filedialog.askdirectory()


if 'linux' in sys.platform:
    import gi
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk, Gdk, GLib

    def _quick_buttons(dialog):
        """
        Add cancel buttons to GTK dialogs.
        """
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL)

    def _open_file():
        """
        GTK backend file dialog for opening a single file.
        """
        result = []

        def _open_file_helper(_None):
            """
            Wrapped file opening file dialog window class.
            """
            dialog = Gtk.FileChooserDialog(action=Gtk.FileChooserAction.OPEN)
            _quick_buttons(dialog)
            dialog.add_button(Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
            dialog.run()
            path = dialog.get_filename()
            dialog.destroy()
            Gtk.main_quit()
            result.append(path)
        Gdk.threads_add_idle(GLib.PRIORITY_DEFAULT, _open_file_helper, None)
        Gtk.main()
        return result[0]

    def _open_files():
        """
        GTK backend file dialog for opening multiple files.
        """
        result = []

        def _open_files_helper(_None):
            """
            Wrapped multiple file opening file dialog window class.
            """
            dialog = Gtk.FileChooserDialog(action=Gtk.FileChooserAction.OPEN)
            dialog.set_select_multiple(True)
            _quick_buttons(dialog)
            dialog.add_button(Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
            dialog.run()
            paths = dialog.get_filenames()
            dialog.destroy()
            Gtk.main_quit()
            result.append(paths)
        Gdk.threads_add_idle(GLib.PRIORITY_DEFAULT, _open_files_helper, None)
        Gtk.main()
        return result[0]

    def _save_file(filters=[], name=''):
        """
        GTK backend file dialog for saving a single file.
        """
        result = []

        def _save_file_helper(filters=[], name=''):
            """
            Wrapped file saving file dialog window class.
            """
            dialog = Gtk.FileChooserDialog(action=Gtk.FileChooserAction.SAVE)
            _quick_buttons(dialog)
            dialog.add_button(Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
            # for f in filters:
            #     filter_text = Gtk.FileFilter()
            #     filter_text.set_name(f[0])
            #     dialog.add_filter(filter_text)
            dialog.run()
            path = os.path.join(dialog.get_current_folder(),
                                dialog.get_current_name())
            dialog.destroy()
            Gtk.main_quit()
            result.append(path)
        Gdk.threads_add_idle(GLib.PRIORITY_DEFAULT, _save_file_helper, None)
        Gtk.main()
        return result[0]

    def _open_folder():
        """
        GTK backend file dialog for opening a folder.
        """
        result = []

        def _open_folder_helper(_None):
            """
            Wrapped folder opening file dialog window class.
            """
            dialog = Gtk.FileChooserDialog(
                action=Gtk.FileChooserAction.SELECT_FOLDER)
            _quick_buttons(dialog)
            dialog.add_button(Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
            dialog.run()
            path = dialog.get_filename()
            dialog.destroy()
            Gtk.main_quit()
            result.append(path)
        Gdk.threads_add_idle(GLib.PRIORITY_DEFAULT, _open_folder_helper, None)
        Gtk.main()
        return result[0]

# Holds utilities that many parts of the peak tracker use.


class Data_File:
    """
    Stores the x, y, f, and v data for an imported tdms file.

    Attributes
    ----------
    x : arr
        1D array of the data's x values.
    y : arr
        1D array of the data's y values.
    f : arr
        1D array of the frequencies data is collected at.
    r : arr
        1D array of data's combined magnitudes. See notes. Identical to `v`.
    v : arr
        1D array of data's combined magnitudes. See notes. Identical to `r`.
    T : float
        1D array of temperatures data is collected at. Defaults to probe
        temperature. If the probe temperature is not available, then defaults
        to cryostat temperature. If the cryostat temperature is also not
        available then the temperature is approximated by interpolating
        between the start and end temperatures imported from the TDMS metadata.
    params : arr
        2D parameters array for Lorentzians fitted to the data file.
    stamp : str
        Metadata imported from corresponding TDMS file.
    date : str
        Date at which TDMS file was created.
    time : str
        Time at which TDMS file was created.
    name : str
        Name of the imported TDMS file.
    probe_temp : arr
        1D array of the temperatures of RUS probe.
    cryo_temp : arr
        1D array of the temperatures of RUS cryostat.
    start_temp : float
        Temperature at start of sweep. Imported from TDMS metadata.
    end_temp : float
        Temperature at end of sweep. Imported from TDMS metadata.

    Notes
    -----
    Here `v` is defined as v = sqrt(x ** 2 + y ** 2). This is identical to `r`.
    """

    def __init__(self, x, y, f):
        self.x = x
        self.y = y
        self.f = f
        self.r = np.sqrt(x ** 2 + y ** 2)
        self.v = self.r
        self.params = None
        self.stamp = None
        self.date = None
        self.time = None
        self.name = None
        self.probe_temp = None
        self.cryo_temp = None
        self.start_temp = None
        self.end_temp = None
        self.T = None

    def _import_probe_temp(self, probe_temp):
        probe_temp = remove_nans(probe_temp)
        min_T = min(probe_temp)
        max_T = max(probe_temp)
        len_T = len(self.f)
        self.probe_temp = normalize_1d(probe_temp, (min_T, max_T, len_T))

    def _import_cryo_temp(self, cryo_temp):
        cryo_temp = remove_nans(cryo_temp)
        min_T = min(cryo_temp)
        max_T = max(cryo_temp)
        len_T = len(self.f)
        self.cryo_temp = normalize_1d(cryo_temp, (min_T, max_T, len_T))

    def _import_meta(self, stamp):
        self.stamp = stamp
        try:
            self.date, self.time, self.start_temp, self.end_temp = stamp.split(
                '_')
            self.name = stamp
        except BaseException:
            self.date, self.time, self.start_temp = stamp.split('_')
            self.end_temp = self.start_temp
            self.name = stamp
        self.start_temp = float(self.start_temp[:-1])
        self.end_temp = float(self.end_temp[:-1])

    def _set_temp(self):
        try:
            probe_high = max(self.probe_temp)
            probe_low = min(self.probe_temp)
            if np.abs(
                    probe_high -
                    1.875) < .03 and np.abs(
                    probe_low -
                    1.875) < .03:
                self.T = self.cryo_temp
            elif any(np.isnan(self.probe_temp)):
                self.T = self.cryo_temp
            else:
                self.T = self.probe_temp
        except BaseException:
            self.T = np.linspace(self.start_temp, self.end_temp, len(self.f))

    def get_temp(self, f0):
        """
        Get temperature corresponding to a particular frequency value. The
        temperature is taken from T, however that is determined.

        Parameters
        ----------
        f0 : float
            Frequency value of temperature.
        """
        index = find_nearest_index(self.f, f0)
        return self.T[index]

    def set_params(self, params):
        """
        Attach Lorentzian parameter array to data file.

        Parameters
        ----------
        params : arr
            2D Lorentzian parameter array.
        """
        self.params = params


def set_all_params(data_files, params):
    """
    Set parameters for many sweeps to all items in a list of data files.

    Parameters
    ----------
    data_files : list
        List of data files. Typically corresponding to a set of continuous
        sweeps imported from the same folder.
    params : arr
        3D parameter array with the same number of elements along axis 0 as
        there are data files in the data_files list.

    Notes
    -----
    Also compatible with 2D parameter arrays as long as a single data file is
    provided instead of a list.
    """
    if type(data_files) is list:
        for i in range(len(data_files)):
            data_files[i].set_params(params[i])
    else:
        data_files.set_params(params)


def get_all_params(data_files):
    """
    Provides a single array for the Lorentzian parameters attached to a list
    of data files.

    Parameters
    ----------
    data_files : list
        List of data files. Typically corresponding to a set of continuous
        sweeps imported from the same folder. Must have parameter attributes
        other than None in order to work properly.
    """
    params = []
    if type(data_files) is list:
        for i in range(len(data_files)):
            params.append(data_files[i].params)
    else:
        params = data_files.params
    try:
        return np.array(params)
    except BaseException:
        return params


def scale_1d(x):
    """
    Determines the scale values for the given 1D array of data.

    Parameters
    ----------
    x : arr
        1D array to get normalization from.

    Returns
    -------
    tuple
        Three element scale tuple corresponding to provided array. The first
        is the minimum value of the array, the second is the maximum value of
        the array, and the last is the total number of indices in the provided
        array.

    Notes
    -----
    Scales are structured as `(min_value, max_value, total_number_of_indices)`.
    """
    return (min(x), max(x), len(x))


def match_lengths(arrs):
    """
    Normalizes a set of arrays around the same scale values.

    Parameters
    ----------
    arrs : list
        List of the arrays to normalize.

    Returns
    -------
    list
        List of the normalized arrays. They are sorted in order of most to
        least elements from the original inputs.

    See Also
    --------
    scale_1d : Method to determine normalization scale.
    """
    return_arrs = arrs.copy()
    arrs.sort(key=lambda a: len(a))
    length = len(arrs[0])
    for i in range(0, len(return_arrs)):
        a = return_arrs[i]
        a = normalize_1d(a, (min(a), max(a), length))
        return_arrs[i] = a
    return return_arrs


def normalize_1d(x, scale=(0, 1, 1024)):
    """
    Normalizes a given array of data around the provided scale factor.

    Parameters
    ----------
    x : arr
        1D array to normalize.
    scale : tuple
        Three element scale tuple.

    Returns
    -------
    arr
        The normalized 1D array.

    Notes
    -----
    Scales are structured as `(min_value, max_value, total_number_of_indices)`.
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


def remove_nans(arr):
    """
    Removes nans from a Lorentzian parameter array of up to two dimensions.

    Parameters
    ----------
    arr : arr
        A 1D or 2D Lorentzian parameter array that may or may non have nans.

    Returns
    -------
    arr
        A new array with all the nans correctly removed.
    """
    if len(arr.shape) == 1:
        return arr[~np.isnan(arr)]
    elif len(arr.shape) == 2:
        new_arr = np.empty((0, len(arr[0])))
        for i in range(0, len(arr)):
            if not any(np.isnan(arr[i])):
                new_arr = np.append(new_arr, np.array([arr[i]]), axis=0)
        return new_arr


def _progressbar(it, prefix="", size=60, file=sys.stdout, progress=True):
    """
    Use with an iteratore as 'it' to show a progress bar while waiting.

    Parameters
    ----------
    it : iterator
        Iterator to build the bar from.
    prefix : str
        What to say before the progress.
    size : int
        Length of bar.
    file : readout
        Where to print the results to.
    progress : bool
        Whether or not to reveal the bar.
    """
    count = len(it)

    def show(j):
        try:
            x = int(size * j / count)
        except BaseException:
            x = 0
        if progress:
            print("%s[%s%s] %i/%i\r\r" %
                  (prefix, "=" * x, "-" * (size - x), j, count), end='\r')
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)


def load_file(path=None):
    """
    Provides a file dialog to search for a single file. Only provides the path
    of what's selected.

    Parameters
    ----------
    path : str, optional
        Optional file path to use. Will only return the file path if provided.

    Returns
    -------
    str
        Path of the chosen file.

    See Also
    --------
    load : General purpose dialog that actually imports the selected binary.
    """
    if path is None:
        path = _open_file()
    return path


def load_files(paths=[]):
    """
    Provides a file dialog to search for multiple files. Only provides the path
    of what's selected.

    Parameters
    ----------
    path : str, optional
        Optional file path to use. Will only return the file path if provided.

    Returns
    -------
    str
        Path of the chosen file.

    See Also
    --------
    load : General purpose dialog that actually imports the selected binary.
    """
    if len(paths) == 0:
        paths = _open_files()
    return paths


def load_dir(path=None):
    """
    Provides a file dialog to search for a directory. Only provides the path of
    what's selected.

    Parameters
    ----------
    path : str, optional
        Optional file path to use. Will only return the file path if provided.

    Returns
    -------
    str
        Path of the chosen file.

    See Also
    --------
    load : General purpose dialog that actually imports the selected binary.
    """
    if path is None:
        path = _open_folder()
        print(path)
    return path


def import_tdms_file(path=None, show=False):
    """
    Import a TDMS file as a Data_File object.

    Parameters
    ----------
    path : str, optional
        File path to import from. A file dialog will open if no path is
        provided.
    show : bool, optional
        If True then prints information about the imported TDMS file.

    Notes
    -----
    This is a wrapper for `_import_file` that automatically processes it after
    importing. Calling `_import_file()` directly is deprecated and will
    likely be removed.
    """
    data_file, path = _import_file(path=path, show=show, include_path=True)
    stamp = os.path.basename(path)[:-5]
    data_file._import_meta(stamp)
    data_file._set_temp()
    return data_file


def _import_file(path=None, show=False, include_path=False):
    """
    Import a TDMS file as a Data_File object.

    Parameters
    ----------
    path : str, optional
        File path to import from. A file dialog will open if no path is
        provided.
    show : bool, optional
        If True then prints information about the imported TDMS file.

    Notes
    -----
    This is deprecated and will be removed. Please use `import_tdms_file()`.
    """
    if path is None:
        path = _open_file()
    tdms_file = TdmsFile.read(path)
    channels = tdms_file.groups()[0].channels()
    tdms_f = remove_nans(channels[0][:])
    tdms_x = remove_nans(channels[1][:])
    tdms_y = remove_nans(channels[2][:])
    [tdms_f, tdms_x, tdms_y] = match_lengths([tdms_f, tdms_x, tdms_y])
    tdms_file = Data_File(tdms_x, tdms_y, tdms_f)
    if len(channels) >= 5:
        tdms_probe_temp = remove_nans(channels[3][:])
        tdms_cryo_temp = remove_nans(channels[4][:])
        tdms_file._import_probe_temp(tdms_probe_temp)
        tdms_file._import_cryo_temp(tdms_cryo_temp)
    if show:
        print('Imported file from ' + str(path))
    if include_path:
        return tdms_file, path
    return tdms_file


def plot_region(
        i,
        regions,
        f,
        v,
        color=None,
        show_boundaries=False,
        min_color='g',
        max_color='g'):
    """
    Given a region array, some frequency data, and some displacement data, will
    plot the specified index.

    Parameters
    ----------
    i : int
        Index.
    regions : arr
        Array of two element frequency regions.
    f : arr
        1D frequency array.
    v : arr
        1D amplitude array.
    color : str, optional
        Color to plot.
    show_boundaries : bool
        Whether or not to show the provided region boundaries.
    min_color : str, optional
        Color to use for the minimum boundary markers.
    max_color : str, optional
        Color to use for the maximum boundary markers.
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
    Given a region array, some frequency data, and some displacement data, will
    return the frequency and displacement for the region specified by an index.

    Parameters
    ----------
    i : int
        Index.
    regions : arr
        Array of two element frequency regions.
    f : arr
        1D frequency array.
    v : arr
        1D amplitude array.

    Returns
    -------
    arr
        All the frequency values within the specified region.
    arr
        All the amplitude values within the specified region.
    """
    min_f = int(regions[i][0])
    max_f = int(regions[i][1])
    region_f = f[min_f:max_f]
    region_v = v[min_f:max_f]
    return region_f, region_v


def _bit_invert(b):
    """
    Inverts the provided bits.

    Parameters
    ----------
    b : array_like
        Array of 1s and 0s.

    Returns
    arr
        Array of 0s and 1s.
    """
    return np.abs(b - 1)


def drop_region(regions, min_length=10, max_length=None):
    """
    Given a region array, will return only the regions with more than the
    min_length number of indices.

    Parameters
    ----------
    regions : arr
        Array of two element frequency regions.
    min_length : int, optional
        Minimum number of indices to use for filtering regions.
    max_length : int, optional
        Maximum number of indices to use for filtering regions. Defaults to
        None and if left that way will have no maximum.
    """
    kept_regions = np.empty((0, 2))
    if max_length is None:
        max_length = np.inf
    for i in range(0, len(regions)):
        region = regions[i]
        if region[1] - region[0] >= min_length and region[1] - \
                region[0] <= max_length:
            kept_regions = np.append(kept_regions, np.array([region]), axis=0)
    return kept_regions


def order_difference(val_1, val_2):
    """
    Returns how many orders of magnitude val_1 and val_2 differ by.

    Parameters
    ----------
    val_1 : float
    val_2 : float

    Returns
    -------
    float
    """
    return np.abs(np.log10(val_1) - np.log10(val_2))


def compare_lorentz(l1, l2, f):
    """
    Given two Lorentzian parameter arrays will provide a numerical value of how
    similar the two Lorentzians are to each other. This is a measure.

    Parameters
    ----------
    l1 : arr
        1D Lorentzian parameter array.
    l2 : arr
        1D Lorentzian parameter array.

    Returns
    -------
    float
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
    Returns the index of the value in an array closest to a provided value.

    Parameters
    ----------
    arr : arr
        1D array to search from.
    val : float
        Value to search for.

    Returns
    -------
    int
    """
    reduced_arr = np.abs(arr - val)
    min_reduced_val = min(reduced_arr)
    return np.where(reduced_arr == min_reduced_val)[0][0]


def _simplify_regions(regions):
    """
    Simplifies set of regions.

    Parameters
    ----------
    regions : arr
        Array of two element frequency regions.

    Returns
    -------
    arr
        The simplified array of two index frequency regions.
    """
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
        simplified_regions = np.append(
            simplified_regions, np.array([[start, end]]), axis=0)
    return simplified_regions


def param_sort(params_2d):
    """
    Sorts a 2D parameter array in order of frequency.

    Parameters
    ----------
    params_2d : arr
        2D Lorentzian parameter array.

    Returns
    -------
    arr
        A 2D Lorentzian parameter array sorted by frequencies.
    """
    not_nan_params = np.empty((0, 4))
    nan_params = np.empty((0, 4))
    nan_indices = []
    for i in range(0, len(params_2d)):
        if any(np.isnan(params_2d[i])):
            nan_indices.append(i)
            nan_params = np.append(nan_params, [params_2d[i]], axis=0)
        else:
            not_nan_params = np.append(not_nan_params, [params_2d[i]], axis=0)
    not_nan_params = not_nan_params[not_nan_params[:, 1].argsort()]
    final_params = np.empty((0, 4))
    offset = 0
    for i in range(0, len(params_2d)):
        if i in nan_indices:
            offset += 1
            final_params = np.append(
                final_params, [[np.nan, np.nan, np.nan, np.nan]], axis=0)
        else:
            final_params = np.append(
                final_params, [not_nan_params[i - offset]], axis=0)
    return final_params


def append_params_3d(p1, p2, force=False):
    """
    Adds new Lorentzians to an existing set of parameters. Does not add later
    data points of existing Lorentzians!

    Parameters
    ----------
    p1 : arr
        3D Lorentzian parameter array.
    p2 : arr
        3D Lorentzian parameter array.
    force : bool, optional
        Determines whether or not to force a merged 3D paramter array should be
        created even if the two provided arrays aren't normally compatible.

    Returns
    -------
    arr
        Combined 3D Lorentzian parameter array.
    """
    p3 = []
    if p2 is None or 0 in np.array(p2).shape:
        return p1
    if p1 is None or 0 in np.array(p1).shape:
        return p2
    if not force:
        if not len(p1) == len(p2):
            raise ValueError("Parameters not of the same length.")
        else:
            for i in range(0, len(p1)):
                combined = np.append(p1[i], p2[i], axis=0)
                p3.append(combined)
    else:
        if len(p1) > len(p2):
            p3 = p1
        else:
            p3 = p2
    return np.array(p3)


def save(some_object, path=None, name=''):
    """
    Saves any python object as a pickled binary.

    Parameters
    ----------
    some_object : obj
        An object to pickle and save as a `.pkl` file.
    path : str, optional
        Where to save the binary to. Will open a file dialog if no path is
        provided.
    name : str
        The name of the file to save. Will blank in file dialog if no name is
        provided.

    Returns
    -------
    str
        The path the object has been saved to.

    See Also
    --------
    load : Function to load objects saved in this way.
    """
    if path is None:
        path = _save_file(filters=(("python objects", "*.pkl"),
                                   ("all files", "*.*")), name=name)
    pickle.dump(some_object, open(path, "wb"))
    return path


def load(path=None):
    """
    Loads any pickled python object from binary form.

    Parameters
    ----------
    path : str, optional
        File path of saved binary to load pickled object from.

    Returns
    -------
    obj
        Whatever the saved object is.

    See Also
    --------
    save : Function to save objects this is capable of loading.
    """
    if path is None:
        path = _open_file()
    return pickle.load(open(path, "rb"))


def import_tdms_files(paths=[], show=True):
    """
    Import multiple TDMS files as a list of data files. This is the recommended
    way to import some, but not all, TDMS files within a given directory.

    Parameters
    ----------
    paths : list
        List of strings for the file paths of all the TDMS files that should be
        imported.
    show : bool, optional
        Whether or not to print information about the imported TDMS files.

    Returns
    -------
    list
        List of the data file objects corresponding to the imported TDMS files.

    See Also
    --------
    Data_File : Definition of a data file object.
    """
    paths = load_files(paths)
    data_files = []
    if show:
        print(paths)
    i = 0
    with tqdm(total=len(paths), file=sys.stdout) as pbar:
        for p in paths:
            pbar.set_description('Imported: %d' % (1 + i))
            pbar.update(1)
            sleep(0.01)
            if p[-5:] == '.tdms':
                stamp = os.path.basename(p)[:-5]
                data_file = _import_file(p)
                data_file._import_meta(stamp)
                data_file._set_temp()
                data_files.append(data_file)
    data_files.sort(key=lambda d: int(str(d.date) + str(d.time)))
    print('Imported file order:')
    for i in range(len(data_files)):
        pre = str(i) + ': ' + \
            ((len(str(len(data_files) - 1)) - len(str(i))) * ' ')
        print(pre + str(data_files[i].stamp))
    return data_files


def import_tdms_dir(path=None, show=True):
    """
    Import multiple TDMS files as a list of data files. This is the recommended
    way to import all TDMS files within a given directory.

    Parameters
    ----------
    paths : list
        List of strings for the file paths of all the TDMS files that should be
        imported.
    show : bool, optional
        Whether or not to print information about the imported TDMS files.

    Returns
    -------
    list
        List of the data file objects corresponding to the imported TDMS files.

    See Also
    --------
    Data_File : Definition of a data file object.
    """
    path = load_dir(path)
    names = os.listdir(path)
    data_files = []
    if show:
        print(path)
    i = 0
    with tqdm(total=len(names), file=sys.stdout) as pbar:
        for name in names:
            pbar.set_description('Imported: %d' % (1 + i))
            pbar.update(1)
            sleep(0.01)
            if name[-5:] == '.tdms':
                stamp = name[:-5]
                file_path = os.path.join(path, name)
                data_file = _import_file(file_path)
                data_file._import_meta(stamp)
                data_file._set_temp()
                data_files.append(data_file)

    data_files.sort(key=lambda d: int(str(d.date) + str(d.time)))
    return data_files


def get_temperatures(data_files):
    """
    Get a temperature array from a list of data files.

    Parameters
    ----------
    data_files : list
        List of data files.

    Returns
    -------
    arr
        Array of the starting temperature for the sweep represented by each
        data file. Has the same number of elements as the provided list of
        data files.
    """
    temperatures = []
    for i in range(0, len(data_files)):
        temperatures.append(float(data_files[i].start_temp))
    return np.array(temperatures)


def get_freqs(data_files):
    """
    Get array of frequencies from the Lorentzians fitted to a list of data
    files.

    Parameters
    ----------
    data_files : list
        List of data files.

    Returns
    -------
    arr
        2D array of the frequencies corresponding to the fitted Lorentzians
        attached to the provided list of data files. Axis 0 determines which
        sweep is being referenced and axis 1 has the frequencies for all the
        peaks fitted within a given sweep. When displayed as a table, the
        columns show the movement of a peak across all the sweeps.

    See Also
    --------
    get_temps_and_freqs : As this but with temperatures as well.
    save_freqs : Save peak frequenceis found this way as a CSV.
    save_freqs_with_temps : Save peak frequencies and temperatures as a CSV.
    """
    p = get_all_params(data_files)
    f = []
    for i in range(0, len(data_files)):
        f.append(p[i][..., 1])
    return freq_sort_2d(np.array(f))


def get_temps_and_freqs(data_files):
    """
    Get array of frequencies and temperatures from the Lorentzians fitted to a
    list of data files.

    Parameters
    ----------
    data_files : list
        List of data files.

    Returns
    -------
    arr
        2D array of the frequencies corresponding to the fitted Lorentzians
        attached to the provided list of data files. Axis 0 determines which
        sweep is being referenced and axis 1 has the frequencies for all the
        peaks fitted within a given sweep. When displayed as a table, the
        columns show the movement of a peak across all the sweeps. In this
        format the temperatures as shown in the first column.

    See Also
    --------
    get_freqs : As this but without temperatures. All data is frequencies.
    save_freqs : Save peak frequencies as a CSV.
    save_freqs_with_temps : Save peak frequencies and temperatures as a CSV.
    """
    f = get_freqs(data_files)
    T = np.transpose([get_temperatures(data_files)])
    fT = np.append(T, f, axis=1)
    return fT


def _matplotlib_mac_fix():
    """
    Fix tkinter backend on older macs.
    """
    if 'mac' in sys.platform:
        import matplotlib
        import importlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        importlib.reload(plt)


def _scatter_pts(pts, ref_arr, tar_arr):
    """
    Tool to create array of points for `Point_Selector` to display as an
    interactive scatter plot.

    Parameters
    ----------
    pts : arr
        1D frequency array of points to plot.
    ref_arr : arr
        1D frequency array.
    tar_arr : arr
        1D amplitude array.

    Returns
    -------
    arr
        Amplitude points to preview in scatter plot.

    See Also
    --------
    plot_points : Method of lf._Point_Selector
    """
    arr = np.empty((0,))
    for i in range(len(pts)):
        j = find_nearest_index(pts[i], ref_arr)
        arr = np.append(arr, [tar_arr[j]])
    return arr


def save_freqs(f, name='freqs_kHz', alter=True):
    """
    Save frequencies as a CSV.

    Parameters
    ----------
    f : arr
        1D array of frequencies to be saved.
    name : str, optional
        Name of the saved CSV.
    alter : bool, optional
        If True, saves as kHz. If False, saves as Hz.
    """
    if alter:
        f = np.sort(f) / 1000
    path = _save_file(
        filters=(
            ("text file", "*.txt"),
            ("all files", "*.*")),
        name=name)
    np.savetxt(path, f, delimiter='\n', fmt='%10.15f')


def save_freqs_with_temps(data_files, name='temp_K_and_freqs_kHz'):
    """
    Save frequencies and temperatures as a CSV. The rows correspond to sweeps
    and the colums correspond to individual peaks. The first column is the
    temperature each sweep was taken at. This saves them from the data files
    directly.

    Parameters
    ----------
    data_files : list
        List of data files.
    name : str, optional
        Name of the CSV.
    """
    T = []
    for i in range(len(data_files)):
        T.append(np.mean(data_files[i].T))
    p = get_all_params(data_files)
    f = p[:, :, 1] / 1000
    arr = np.append(np.transpose([T]), f, axis=1)
    arr = temp_freq_sort_2d(arr)
    path = _save_file(
        filters=(
            ("text file", "*.txt"),
            ("comma separated values", "*.csv"),
            ("all files", "*.*")),
        name=name)
    np.savetxt(path, arr, delimiter=',', fmt='%10.15f')


def save_Tf(Tf, path=None, name='temp_K_and_freqs_kHz'):
    """
    Save frequencies and temperatures as a CSV. The rows correspond to sweeps
    and the colums correspond to individual peaks. The first column is the
    temperature each sweep was taken at. This saves them from as array of
    temperatures and frequencies.

    Parameters
    ----------
    Tf : arr
        2D array of temperatures and frequencies.
    path : str, optional
        File path for saved CSV. Loads a file dialog if no path is provided.
    name : str, optional
        Name to use for saved CSV.

    See Also
    --------
    get_temps_and_freqs : Function to get array used for `save_Tf`.
    """
    if path is None:
        path = _save_file(
            filters=(
                ("text file", "*.txt"),
                ("comma separated values", "*.csv"),
                ("all files", "*.*")),
            name=name)
    T = np.transpose(np.array([Tf[:, 0]]))
    f = Tf[:, 1:] / 1000
    Tf = np.append(T, f, axis=1)
    np.savetxt(path, Tf, delimiter=',', fmt='%10.15f')


def load_freqs(path=None):
    """
    Load a 1D array of frequencies. Assumes that they are saved as kHz.

    Parameters
    ----------
    path : str, optional
        File path to load from. Loads a file dialog if no path is provided.

    See Also
    --------
    save_freqs : Function to save frequencies loaded by `load_freqs`.
    """
    if path is None:
        path = _open_file()
    f = np.loadtxt(path, delimiter='\n')
    return f * 1000


def load_freqs_with_temps(path=None):
    """
    Loads temperatures and frequencies as an array.

    Parameters
    ----------
    path : str, optional
        File path to load from. Loads a file dialog if no path is provided.

    Returns
    -------
    arr
        2D array of temperatures and frequencies.

    See Also
    --------
    save_Tf :
        Function to save CSVs usable by `load_freqs_with_temps` starting with
        the same format that it creates.
    save_freqs_with_temps :
        Function to save CSVs usable by `load_freqs_with_temps` starting with
        data files.
    """
    if path is None:
        path = _open_file()
    Tf = np.loadtxt(path, delimiter=',')
    f = np.transpose(Tf)[1:] * 1000
    T = np.array([np.transpose(Tf)[0]])
    Tf = np.append(T, f, axis=0)
    return np.transpose(Tf)


def attach_temps_to_parameters(data_files):
    """
    Makes extended parameter array that includes the temperatures they were
    measured at.

    Parameters
    ----------
    data_files : list
        List of data files.

    Returns
    -------
    arr
        Extended parameter array.
    """
    p = get_all_params(data_files)
    q = []
    for i in range(len(p)):
        q.append([])
        for j in range(len(p[i])):
            p0 = p[i][j]
            f0 = p0[1]
            if np.isnan(f0):
                T0 = np.nan
            else:
                index = find_nearest_index(data_files[i].f, f0)
                T0 = data_files[i].T[index]
            p0 = np.append(p0, [T0])
            q[i].append(p0)
    return np.array(q)


def delete_parameters(params_3d, index):
    """
    Given a 3D Lorentzian parameter array, removes a specific sweep as
    specified by index.

    Parameters
    ----------
    params_3d : arr
        3D Lorentzian parameter array.
    index : int
        Index to remove.

    Returns
    -------
    arr
        3D Lorentzian parameter array with the given sweep removed.
    """
    p = []
    for i in range(len(params_3d)):
        p.append([])
        for j in range(len(params_3d[i])):
            if not j == index:
                p[i].append(params_3d[i][j])
    return np.array(p)


def delete_parameters_from_f_regions_3d(parameters_3d, f_regions):
    """
    Removes the parameters within specific regions from a 3D Lorentzian
    parameter array.

    Parameters
    ----------
    parameters_3d : arr
        3D Lorentzian parameter array to remove Lorentzians from.
    f_regions : arr
        Array of two element frequency regions.

    Returns
    -------
    arr
        3D Lorentzian parameter array with the given Lorentzians removed.
    """
    p = []
    for i in range(len(parameters_3d)):
        p_2d = delete_parameters_from_f_regions_2d(
            parameters_3d[i], f_regions[i])
        p.append(p_2d)
    return np.array(p)


def delete_parameters_from_f_regions_2d(parameters_2d, f_region):
    """
    Removes the parameters within specific regions from a 2D Lorentzian
    parameter array.

    Parameters
    ----------
    parameters_2d : arr
        2D Lorentzian parameter array to remove Lorentzians from.
    f_regions : arr
        Array of two element frequency regions.

    Returns
    -------
    arr
        2D Lorentzian parameter array with the given Lorentzians removed.
    """
    p = []
    for i in range(len(parameters_2d)):
        if (parameters_2d[i][1] >= f_region[0]) and (parameters_2d[i][1] <=
            f_region[1]):
            p.append([np.nan, np.nan, np.nan, np.nan])
        else:
            p.append(parameters_2d[i])
    return np.array(p)


def freq_sort_2d(f_2d):
    """
    Sort a 2D frequency array. This would likely be an array specifically
    containing peaks as found in `get_freqs`. Axis 0 corresponds to the sweeps
    and axis 1 corresponds to the frequencies within each sweep. This sorts
    the peaks for all sweeps with respect to the mean value of that peak's
    location over all sweeps. So, there may be certain sweeps where the peaks
    appear to be out of order. But, that only happens because several peaks
    cross each other due to a change in temperature, pressure, or under some
    field.

    Parameters
    ----------
    f_2d : arr
        2D frequency array.

    Returns
    -------
    arr
        Sorted 2D frequency array.

    See Also
    --------
    get_freqs : Function to get 2D frequency arrays.
    """
    f_2d = np.transpose(f_2d)
    final_sorting_order = []
    for i in range(len(f_2d)):
        mean_freq = np.mean(remove_nans(f_2d[i]))
        final_sorting_order.append((i, mean_freq))
    # print(np.array(final_sorting_order))
    final_sorting_order.sort(key=lambda t: t[1])
    # print(np.array(final_sorting_order))
    sorted_freqs = []
    for i in range(len(f_2d)):
        sorted_freqs.append(f_2d[final_sorting_order[i][0]])
    sorted_freqs = np.transpose(np.array(sorted_freqs))
    return sorted_freqs


def temp_freq_sort_2d(Tf_2d):
    """
    As `freq_sort_2d` but with compatibility for the first element within each
    sweep being the starting temperature of that sweep.

    Parameters
    ----------
    Tf_2d : arr
        2D temperature and frequency array.

    Returns
    -------
    arr
        Sorted 2D temperature and frequency array.

    See Also
    --------
    freq_sort_2d : Function to sort just frequencies.
    get_temps_and_freqs : Function to get arrays that would be sorted.
    """
    T = np.transpose(Tf_2d)[0]
    f = np.transpose(Tf_2d)[1:]
    f = freq_sort_2d(np.transpose(f))
    Tf = np.append([T], np.transpose(f), axis=0)
    Tf = np.transpose(Tf)
    return Tf


def find_missing_3d(params_3d, overlap=1, degree=2, sanity_check=True):
    """
    Fills in the nans in the input array. For a given section nans, finds a
    polynomial fit over the region both before and after that region. Then
    uses a transformed sinusoidal connection function to smoothly transition
    between those two polynomials and project the values into the nans. This
    works best for input arrays of smoothly changing functions. Nans at the
    ends are not filled in.

    Parameters
    ----------
    params_3d : arr
        A 3d Lorentzian parameter array.
    overlap : int, optional
        Number of indices above and below to consider, by default 1.
    degree : int, optional
        Maximum degree of polynomial to fit, by default 2. Note that the
        degree is lowered depending on the numbed of points available to
        produce the fit so that there is a meaningful unique fit in each case.

    Returns
    -------
    arr
        The array with the nans filled in.
    """
    new_list = []
    for i in range(params_3d.shape[1]):
        p = params_3d[:, i]
        new_list.append(
            find_missing_2d(
                p,
                overlap=overlap,
                degree=degree,
                sanity_check=sanity_check))
    new_misshapen = np.array(new_list)
    new_params = []
    for i in range(params_3d.shape[0]):
        new_params.append(new_misshapen[:, i])
    return np.array(new_params)


def find_missing_2d(params_2d, overlap=1, degree=2, sanity_check=True):
    """
    Fills in the nans in the input array. For a given section nans, finds a
    polynomial fit over the region both before and after that region. Then
    uses a transformed sinusoidal connection function to smoothly transition
    between those two polynomials and project the values into the nans. This
    works best for input arrays of smoothly changing functions. Nans at the
    ends are not filled in.

    Parameters
    ----------
    params_2d : arr
        A 2d Lorentzian parameter array.
    overlap : int, optional
        Number of indices above and below to consider, by default 1.
    degree : int, optional
        Maximum degree of polynomial to fit, by default 2. Note that the
        degree is lowered depending on the numbed of points available to
        produce the fit so that there is a meaningful unique fit in each case.

    Returns
    -------
    arr
        The array with the nans filled in.
    """
    A = find_missing_1d(params_2d[..., 0], overlap=overlap, degree=degree,
                        sanity_check=sanity_check)
    f0 = find_missing_1d(params_2d[..., 1], overlap=overlap, degree=degree,
                         sanity_check=sanity_check)
    FWHM = find_missing_1d(params_2d[..., 2], overlap=overlap, degree=degree,
                           sanity_check=sanity_check)
    phase = find_missing_1d(params_2d[..., 3], overlap=overlap, degree=degree,
                            sanity_check=sanity_check)
    p = np.transpose(np.array([A, f0, FWHM, phase]))
    return p


def find_missing_1d(params_1d, overlap=1, degree=2, sanity_check=True):
    """
    Fills in the nans in the input array. For a given section nans, finds a
    polynomial fit over the region both before and after that region. Then
    uses a transformed sinusoidal connection function to smoothly transition
    between those two polynomials and project the values into the nans. This
    works best for input arrays of smoothly changing functions. Nans at the
    ends are not filled in.

    Parameters
    ----------
    params_1d : arr
        A 1d array.
    overlap : int, optional
        Number of indices above and below to consider, by default 1.
    degree : int, optional
        Maximum degree of polynomial to fit, by default 2. Note that the
        degree is lowered depending on the numbed of points available to
        produce the fit so that there is a meaningful unique fit in each case.

    Returns
    -------
    arr
        The array with the nans filled in.
    """
    cuts = _make_cuts(params_1d)
    mod_params = list(params_1d)
    fillings = []
    for cut in cuts:
        start = _search_down(params_1d, cut[0] - 1, overlap)
        end = _search_up(params_1d, cut[-1] + 1, overlap)
        filling = [start, end]
        fillings.append(filling)
    for i in range(len(cuts)):
        c = cuts[i]
        f = fillings[i]
        start_check = np.abs(c[0] - f[0]) > 1
        end_check = np.abs(c[-1] - f[-1]) > 1
        if start_check and end_check:
            original_piece = params_1d[f[0]:f[-1] + 1]
            param_piece = _filling_function(original_piece, degree=degree)
            allow = not any(param_piece > max(original_piece))
            allow = allow and not any(param_piece < min(original_piece))
            if degree == 0 or (sanity_check and (not allow)):
                p0 = params_1d[c[0] - 1]
                p1 = params_1d[c[-1] + 1]
                param_piece = np.linspace(p0, p1, f[-1] - f[0] + 1)
        elif not (c[0] == f[0] or c[-1] == f[-1]):
            p0 = params_1d[c[0] - 1]
            p1 = params_1d[c[-1] + 1]
            param_piece = np.linspace(p0, p1, f[-1] - f[0] + 1)
        else:
            param_piece = np.full((f[-1] - f[0] + 1), np.nan)
        for j in range(len(param_piece)):
            j_original = j + f[0]
            if not np.isnan(
                    param_piece[j]) and np.isnan(
                    mod_params[j_original]):
                mod_params[j_original] = param_piece[j]
    return np.array(mod_params)


def _make_cuts(params_1d):
    """
    Makes a list of lists specifying the nans in the given array. If the a
    given sublist has two elements then they specify the first and last indices
    that are nans within the corresponding sub-array of nans. If the sublist
    has only one index then the corresponding sub-array is only one element
    long but that element is still a nan.

    Parameters
    ----------
    params_1d : arr
        Array to return the nan cuts of.

    Returns
    -------
    list
        The list of lists.
    """
    indices = np.arange(len(params_1d))
    indices_to_replace = indices[np.isnan(params_1d)]
    try:
        cuts = [[indices_to_replace[0]]]
        for i in range(len(indices_to_replace) - 1):
            if indices_to_replace[i + 1] - indices_to_replace[i] == 1:
                cuts[-1].append(indices_to_replace[i + 1])
            else:
                cuts.append([indices_to_replace[i + 1]])
        if len(cuts[-1]) == 0:
            cuts = cuts[:-1]
    except:
        cuts = []
    return cuts


def _search_up(arr, start, count):
    """
    Searches upwards from the start index until it hits a nan. Returns the
    highest non-nan index within the specified search distance.

    Parameters
    ----------
    arr : arr
        Array to search.
    start : int
        Index to start search from.
    count : int
        Maximum distance to search.

    Returns
    -------
    int
        Found index.
    """
    top = len(arr) - 1
    a = arr[start + 1:]
    step = 0
    actual_steps = 0
    while step < count:
        step += 1
        if len(a) > 0 and not np.isnan(a[0]):
            actual_steps += 1
            a = a[1:]
    return min(top, start + actual_steps)


def _search_down(arr, start, count):
    """
    Searches downwards from the start index until it hits a nan. Returns the
    lowest non-nan index within the specified search distance.

    Parameters
    ----------
    arr : arr
        Array to search.
    start : int
        Index to start search from.
    count : int
        Maximum distance to search.

    Returns
    -------
    int
        Found index.
    """
    a = arr[:start]
    step = 0
    actual_steps = 0
    while step < count:
        step += 1
        if len(a) > 0 and not np.isnan(a[-1]):
            actual_steps += 1
            a = a[:-1]
    return max(0, start - actual_steps)


def _filling_connection_function(x):
    """
    Helper function for `_filling_weight_function` to smoothly connect from 0
    to 1.

    Parameters
    ----------
    x : arr
        Array of x values.

    Returns
    -------
    arr
        Array of C(x) values for connection function C(x).

    See Also
    --------
    _filling_function : Function that this feeds into.
    """
    return np.sin(np.pi * (x - 1 / 2)) / 2 + 1 / 2


def _filling_weight_function(x, x1, x2):
    """
    Function that smoothly fills from 0 to 1 between x1 and x2.

    Parameters
    ----------
    x : arr
        Array of x values.
    x1 : float
        Starting value to fill between.
    x2 : float
        Ending value to fill between.
    """
    out = []
    for x0 in x:
        if x0 <= x1:
            out.append(0)
        elif x0 >= x2:
            out.append(1)
        else:
            out.append(_filling_connection_function((x0 - x1) / (x2 - x1)))
    return np.array(out)


def _filling_combination_function(x, x1, x2, f1, f2):
    """
    Function that smoothly connects polynomials f1 and f2 from x1 to x2.

    Parameters
    ----------
    x : arr
        Domain.
    x1 : float
        Starting value.
    x2 : float
        Ending value.
    f1 : arr
        Coefficients for polynomial representation of starting function.
    f2 : [type]
        Coefficients for polynomial representation of ending funciton.

    Returns
    -------
    arr
        Output with functions connected.
    """
    w = _filling_weight_function

    def g1(x):
        x_outs = []
        for x0 in x:
            x_out = 0
            for i in range(len(f1)):
                x_out += f1[i] * x0 ** (len(f1) - i - 1)
            x_outs.append(x_out)
        return x_outs

    def g2(x):
        x_outs = []
        for x0 in x:
            x_out = 0
            for i in range(len(f2)):
                x_out += f2[i] * x0 ** (len(f2) - i - 1)
            x_outs.append(x_out)
        return x_outs
    return w(x, x1, x2) * g1(x) + (1 - w(x, x1, x2)) * g2(x)


def _filling_function(y, degree=2):
    """
    Fit over the nan gap.

    Parameters
    ----------
    y : arr
        All values of array including nans to be filled between.
    degree : int, optional
        Degree of polynomial function to fit, by default 1. The degree will
        automatically be reduced if there are not enough points provided to
        make a meaningful unique fit.

    Returns
    -------
    arr
        The original y input but with all the nans filled in.
    """
    if np.isnan(y[0]) or np.isnan(y[-1]):
        return y
    else:
        i = 0
        while not np.isnan(y[i]):
            i += 1
        y_start = y[:i]
        j = len(y) - 1
        while not np.isnan(y[j - 1]):
            j -= 1
        y_end = y[j:]
        d1 = min(degree, len(y_start) - 1)
        d2 = min(degree, len(y_end) - 1)
        x_start = np.arange(len(y_start))
        x_end = np.arange(len(y_end)) + j
        fit_start = np.polyfit(x_start, y_start, d1)
        fit_end = np.polyfit(x_end, y_end, d2)
        x_mid = np.arange(j - i) + i
        y_mid = _filling_combination_function(
            x_mid, i - 1, j, fit_start, fit_end)
        return np.concatenate([y_start, y_mid, y_end])


def install_models():
    """
    Install pre-made Lorentzian machine learning models.
    """
    command_string = 'pip3 install git+https://github.com/GabePoel/'
    'Lorentzian-Models#egg=lorentzian_models --user'
    os.system(command_string)
