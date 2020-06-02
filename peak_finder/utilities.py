import sys
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from nptdms import TdmsFile

# Holds utilities that many parts of the peak tracker use.

class DataFile:
    """
    Stores the x, y, f, and v data for an imported tdms file.
    Here v is defined as v = sqrt(x ** 2 + y ** 2).
    """
    def __init__(self, x, y, f):
        self.x = x
        self.y = y
        self.f = f
        self.v = np.sqrt(x ** 2 + y ** 2)

def progressbar(it, prefix="", size=60, file=sys.stdout, progress=True):
    """Use with an iteratore as 'it' to show a progress bar while waiting."""
    count = len(it)
    def show(j):
        x = int(size*j/count)
        if progress:
            file.write("%s[%s%s] %i/%i\r\r" % (prefix, "="*x, "-"*(size-x), j, count))
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)

def import_file(path=None):
    """
    Import a tdms file. Returns a 
    Leave path blank to open a file dialog window and select the file manually. Otherwise pass in a path.
    """
    if path is None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename()
    tdmsFile = TdmsFile(path)
    tdms_f = tdmsFile.object('Untitled', 'freq (Hz)').data
    tdms_x = tdmsFile.object('Untitled', 'X1 (V)').data
    tdms_y = tdmsFile.object('Untitled', 'Y1 (V)').data
    return DataFile(tdms_x, tdms_y, tdms_f)

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

def drop_region(regions, min_length=10):
    """
    Given a region array, will return only the regions with more than the min_length number of indices.
    """
    kept_regions = np.empty((0, 2))
    for i in range(0, len(regions)):
        region = regions[i]
        if region[1] - region[0] >= min_length:
            kept_regions = np.append(kept_regions, np.array([region]), axis=0)
    return kept_regions

def order_difference(val_1, val_2):
    """
    Returns how many orders of magnitude val_1 and val_2 differ by.
    """
    return np.abs(np.log10(val_1) - np.log10(val_2))