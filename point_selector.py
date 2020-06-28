import numpy as np

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

class SelectFromCollection:
    def __init__(self, ax, data, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.alpha_other = alpha_other