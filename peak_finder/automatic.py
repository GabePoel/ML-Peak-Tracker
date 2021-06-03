import numpy as np

from . import sliding_window as sw
from . import fit_lorentz as fl
from . import live_fitting as lf
from . import models

def quick_analyze(f, v, show=True, learn=True):
    """
    Automatically get parameters through a pre-made machine learning model
    and script.

    Parameters
    ----------
    f : arr
        1D frequency array.
    v : arr
        1D amplitude array.

    Returns
    -------
    parameters : arr
        2D Lorentzian parameter array.
    """
    if learn:
        tight_model = models.tight_lorentzian()
        wide_model = models.wide_lorentzian()
        wide_regions = sw.slide_scale(
            wide_model,
            v,
            min_zoom=5,
            max_zoom=7,
            confidence_tolerance=0.95,
            merge_tolerance=0,
            target=1,
            compress=False,
            simplify=False)
        tight_regions = sw.split_peaks(
            tight_model,
            f,
            v,
            wide_regions,
            min_zoom=2,
            max_zoom=7,
            confidence_tolerance=0.0,
            single_zoom=False)
        noise_level = 3 * sw.extract_noise(v)
        parameters = fl.parameters_from_regions(
            f, v, tight_regions, noise_filter=noise_level,
            catch_degeneracies=True)
    else:
        parameters = np.empty((0, 4))
    if show:
        live = lf.Live_Instance(f, v)
        live.import_lorentzians(parameters)
        live.activate()
        parameters = live.get_all_params()
    return parameters
