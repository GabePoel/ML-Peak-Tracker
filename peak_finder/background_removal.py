"""
Tools to remove the background from real data. All are very experimental.
"""

import numpy as np
import PySimpleGUI as sg

from . import automatic as auto
from . import utilities as util
from . import models
from . import generate_lorentz as gl
from . import fit_lorentz as fl
from . import sliding_window as sw
from . import live_fitting as lf
from . import classify_data as cd

def remove_background(data_files, start_index=0):
    """
    Experimental function to remove background noise and Lorentzians from
    data with smaller peaks in it.

    Parameters
    ----------
    data_files : list
    start_index : int, optional

    Returns
    -------
    f_list : list
    v_list : list
    """
    model = models.tight_lorentzian()
    f_list = []
    v_list = []
    # start_file = data_files[start_index]
    # f = start_file.f
    # v = start_file.r
    # coef_1 = np.polyfit(f, v, 2)
    # bg_1 = np.poly1d(coef_1)
    # v = v - bg_1(f)
    # start_params = auto.quick_analyze(f, v, learn=False, show=True)
    # coef_2 = np.polyfit(f, v - gl.multi_lorentz_2d(f, start_params), 2)
    # bg_2 = np.poly1d(coef_2)
    # v = v - bg_2(f)
    # FWHM = start_params[:,2]
    # min_FWHM = min(FWHM)
    # max_FWHM = max(FWHM)
    # delta_f = max(f) - min(f)
    # min_zoom = int(np.floor(np.log2(delta_f / max_FWHM))) - 1
    min_zoom = 1
    # max_zoom = int(np.ceil(np.log2(delta_f / min_FWHM)))
    max_zoom = 5
    for i in util.progressbar(
            range(
                0,
                len(data_files)),
            'Removing Background: '):
        sg.one_line_progress_meter(
            'Background Removeal Progress', i, len(data_files), '-key-')
        # try:
        #     live.close_window()
        # except:
        #     pass
        f = data_files[i].f
        v = data_files[i].r
        # live = lf.Live_Instance(f, v)
        # live.activate()
        coef_1 = np.polyfit(f, v, 2)
        bg_1 = np.poly1d(coef_1)
        v = v - bg_1(f)
        # live = lf.Live_Instance(f, v)
        # live.activate()
        regions = sw.slide_scale(
            model, v, min_zoom=min_zoom, max_zoom=max_zoom)
        while len(regions) > 0:
            bg_params = fl.parameters_from_regions(f, v, regions)
            if len(bg_params) > 0:
                region_deltas = []
                for i in range(0, len(regions)):
                    max_ind = int(regions[i][1])
                    min_ind = int(regions[i][0])
                    max_f = f[max_ind]
                    min_f = f[min_ind]
                    region_deltas.append(max_f - min_f)
                max_delta = max(region_deltas)
                bg_params = np.array(
                    [bg_params[bg_params[:, 2].argsort()][-1]])
                if util.order_difference(bg_params[0][2], max_delta) < 1:
                    # live = lf.Live_Instance(f, v)
                    # live.import_lorentzians(bg_params)
                    # live.activate()
                    if len(bg_params) > 0:
                        v_to_remove = gl.multi_lorentz_2d(f, bg_params)
                        # v_to_remove = cd.normalize_1d(v_to_remove, (min(v), max(v), len(v)))
                        v = v - v_to_remove
                        coef_2 = np.polyfit(f, v, 3)
                        bg_2 = np.poly1d(coef_2)
                        v = v - bg_2(f)
                    regions = sw.slide_scale(
                        model, v, min_zoom=min_zoom, max_zoom=max_zoom)
                else:
                    regions = np.empty((0, 2))
            else:
                regions = np.empty((0, 2))
        # live = lf.Live_Instance(f, v)
        # live.activate()
        # if preview:
        #     live.close_window()
        #     live = lf.Live_Instance(f, v)
        #     live.activate(loop=False)
        f_list.append(f)
        v_list.append(v)
    sg.one_line_progress_meter_cancel('-key-')
    return f_list, v_list
