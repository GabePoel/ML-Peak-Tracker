import sys
import time
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from nptdms import TdmsFile

# Holds utilities that many parts of the peak tracker use.

def progressbar(it, prefix="", size=60, file=sys.stdout, progress=True):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        if progress:
            file.write("%s[%s%s] %i/%i\r\r" % (prefix, "#"*x, "."*(size-x), j, count))
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)

def import_file():
    root = tk.Tk()
    root.withdraw()
    fp = filedialog.askopenfilename()
    tdmsFile = TdmsFile(fp)
    tdms_f = tdmsFile.object('Untitled', 'freq (Hz)').data
    tdms_x = tdmsFile.object('Untitled', 'X1 (V)').data
    tdms_y = tdmsFile.object('Untitled', 'Y1 (V)').data
    tdms_v = np.sqrt(tdms_x ** 2 + tdms_y ** 2)
    return tdms_f, tdms_y

def plot_region(i, regions, f, v, color=None, show_boundaries=False, min_color='g', max_color='g'):
    min_f = int(regions[i][0])
    max_f = int(regions[i][1])
    if color is None:
        plt.plot(f[min_f:max_f], v[min_f:max_f])
    else:
        plt.plot(f[min_f:max_f], v[min_f:max_f], color=color)
    if show_boundaries:
        plt.axvline(x=f[min_f], color=min_color)
        plt.axvline(x=f[max_f], color=max_color)