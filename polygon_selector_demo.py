"""
=====================
Polygon Selector Demo
=====================

Shows how one can select indices of a polygon interactively.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

import importlib
import nptdms
import pickle

from os import path

from peak_finder import utilities as util
from peak_finder import generate_lorentz as gl
from peak_finder import generate_data as gd
from peak_finder import classify_data as cd
from peak_finder import efficient_data_generation as ed
from peak_finder import data_spy as ds
from peak_finder import sliding_window as sw
from peak_finder import train_model as tm
from peak_finder import fit_lorentz as fl
from peak_finder import live_fitting as lf
from peak_finder import track_peaks as tp
from peak_finder import background_removal as br
from peak_finder import automatic
from peak_finder import models

importlib.reload(util)
importlib.reload(gl)
importlib.reload(gd)
importlib.reload(cd)
importlib.reload(ed)
importlib.reload(ds)
importlib.reload(sw)
importlib.reload(tm)
importlib.reload(fl)
importlib.reload(lf)
importlib.reload(tp)
importlib.reload(br)
importlib.reload(automatic)
importlib.reload(models)

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path


class SelectFromCollection:
    """Select indices from a matplotlib collection using `PolygonSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.poly = PolygonSelector(ax, self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    data_files = tp.import_tdms_files()
    print(str(len(data_files)) + ' files imported')
    smol_files = data_files[0:200]

    max_res = 100
    z = np.empty((0, max_res))
    x = np.empty((0,))
    y = np.empty((0,))
    for i in range(0, len(smol_files)):
        z0 = smol_files[i].r
        z0 = cd.scale_zoom(z0, 0, 1)
        z0 = cd.normalize_1d(z0, (min(z0), max(z0), max_res))
        coef = np.polyfit(np.arange(max_res), z0, 3)
        bg = np.poly1d(coef)
        z0 -= bg(np.arange(0, max_res))
        z = np.append(z, [z0], axis=0)
        x0 = np.arange(max_res)
        x = np.append(x, x0)
        y = np.append(y, np.ones(x0.shape) * i)

    fig, ax = plt.subplots()
    pts = ax.scatter(x, y)
    ax.pcolormesh(z, cmap='cool')

    selector = SelectFromCollection(ax, pts)

    print("Select points in the figure by enclosing them within a polygon.")
    print("Press the 'esc' key to start a new polygon.")
    print("Try holding the 'shift' key to move all of the vertices.")
    print("Try holding the 'ctrl' key to move a single vertex.")

    plt.show()

    selector.disconnect()

    # After figure is closed print the coordinates of the selected points
    print('\nSelected points:')
    print(selector.xys[selector.ind])
