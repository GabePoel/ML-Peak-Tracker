__all__ = [
    'preview',
    'color_selection',
    'mistake_selection',
    'live_selection',
    'point_selection',
    '_Live_Instance']

import tkinter as tk
from tkinter import (ttk, simpledialog)

import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.widgets import (
    RectangleSelector,
    PolygonSelector,
    Cursor,
    LassoSelector)
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from .. import classify_data as cd
from .. import fit_lorentz as fl
from .. import generate_lorentz as gl
from .. import utilities as util

try:
    from .. import automatic as auto
    _can_ml = True
except BaseException:
    from .. import automatic_no_ml as auto
    _can_ml = False

util._matplotlib_mac_fix()

_option_colors = {
    "Cool": "cool",
    "Viridis": "viridis",
    "Plasma": "plasma",
    "Inferno": "inferno",
    "Magma": "magma",
    "Cividis": "cividis",
    "Gray": "gray",
    "Bone": "bone",
    "Pink": "pink",
    "Spring": "spring",
    "Summer": "summer",
    "Autumn": "autumn",
    "Winter": "winter",
    "Wistia": "Wistia",
    "Hot": "hot",
    "Copper": "copper",
    "Spectral": "Spectral",
    "Twilight": "twilight",
    "Twilight Shifted": "twilight_shifted",
    "HSV": "hsv",
    "Ocean": "ocean",
    "GIST Earth": "gist_earth",
    "Terrain": "terrain",
    "GIST Stern": "gist_stern",
    "GNU Plot": "gnuplot",
    "GNU Plot 2": "gnuplot2",
    "CMR Map": "CMRmap",
    "Cube Helix": "cubehelix",
    "Blue Red Green": "brg",
    "GIST Rainbow": "gist_rainbow",
    "Rainbow": "rainbow",
    "Jet": "jet"}

_option_select_colors = {
    "Blue": "tab:blue",
    "Orange": "tab:orange",
    "Green": "tab:green",
    "Red": "tab:red",
    "Purple": "tab:purple",
    "Brown": "tab:brown",
    "Pink": "tab:pink",
    "Grey": "tab:grey",
    "Olive": "tab:olive",
    "Cyan": "tab:cyan",
    "Black": "black",
    "White": "white"}


def _lin(x, a, b):
    """
    Definition of a linear function.

    Parameters
    ----------
    x : arr
        The values to act on.
    a : float
        Y intercept.
    b : float
        Slope.

    Returns
    -------
    arr
        The with the function acted on them.
    """
    return x * b + a


def _callback(input):
    """
    Ensures that the inputs for certain tkinter boxes must be integers.

    Parameters
    ----------
    input : str
        The character being inputted.

    Returns
    -------
    bool
        Whether the input character is an acceptable numeric value.
    """
    if input.isdigit():
        return True

    elif input is "":
        return True

    else:
        return False


def preview(f, v, params):
    """
    Look at a given data set with frequencies, amplitudes, and some set of
    already defined parameters.

    Parameters
    ----------
    f : arr
        Frequency array.
    v : arr
        Amplitude array.
    params : arr
        3D Lorentzian parameters array.
    """
    live = _Live_Instance(f, v)
    live.import_lorentzians(params)
    live.activate()


class _Tooltip:
    """
    Tooltip recipe from
    http://www.voidspace.org.uk/python/weblog/arch_d7_2006_07_01.shtml#e387.
    This is nearly the same recipe used in matplotlib menus.
    """
    @staticmethod
    def create_tooltip(widget, text):
        toolTip = _Tooltip(widget)

        def enter(event):
            toolTip.showtip(text)

        def leave(event):
            toolTip.hidetip()
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        """Display text in tooltip window."""
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + self.widget.winfo_width()
        y = y + self.widget.winfo_rooty()
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        try:
            # For Mac OS
            tw.tk.call("::tk::unsupported::MacWindowStyle",
                       "style", tw._w,
                       "help", "noActivates")
        except tk.TclError:
            pass
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         relief=tk.SOLID, borderwidth=1)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


class _Live_Lorentz():
    """
    Class form of Lorentzian data structure to be actively manipulated.
    """

    def __init__(self, p):
        """
        Parameters
        ----------
        p : arr
            1D Lorentzian parameters array.
        """
        self.A = p[0]
        self.f0 = p[1]
        self.FWHM = p[2]
        self.phase = p[3]

    def params(self):
        """
        Returns
        -------
        arr
            1D Lorentzian parameters array.
        """
        return np.array([self.A, self.f0, self.FWHM, self.phase])


class _Selection():
    """
    Selected component of the current window.
    """

    def __init__(self, x_delta, y_delta, x_pos, y_pos):
        """
        Create the instance of the `_Selection` class.

        Parameters
        ----------
        x_delta : int
            Width of the selected window.
        y_delta : int
            Height of the selected window.
        x_pos : int
            The x coordinate of the lower left corner of the selected window.
        y_pos : int
            The y coordinate of the lower left corner of the selected window.
        """
        self.x_delta = x_delta
        self.y_delta = y_delta
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.f_min = self.x_pos
        self.f_max = self.x_pos + self.x_delta


class _Live_Instance():
    """
    The class containing all the information for the single temperature
    interactive Lorentzian window.

    See Also
    --------
    `live_selection`
        Wrapper function that calls upon this class. Includes all keyboard
        shortcuts relevant to this window.
    """

    def __init__(self, f, v, method='lm'):
        self.r = None
        self.y = None
        self.x = None
        self.f = f[np.logical_not(np.isnan(f))]
        self.v = v[np.logical_not(np.isnan(v))]
        self.live_lorentzians = {}
        self.all_data = False
        self.show_components = False
        self.component_height = 1.5
        self.projection_height = -1
        self.method = method
        self.vlines = []

    def import_all_data(self, x, y, data_to_analyze=None):
        """
        Set the provided data to be used. This allows for selecting whether
        just the x or y components of the Lorentzians should be plotted or if
        we should instead use the combiend r = √(x² + y²).

        Parameters
        ----------
        x : arr
            1D array of x amplitude data.
        y : arr
            1D array of the y amplitude data.
        data_to_analyze : [type], optional
            [description], by default None
        """
        self.x = x[np.logical_not(np.isnan(x))]
        self.y = y[np.logical_not(np.isnan(y))]
        r = x ** 2 + y ** 2
        self.r = r[np.logical_not(np.isnan(r))]
        self.all_data = True
        if data_to_analyze == 'r':
            self.v = self.r
        elif data_to_analyze == 'x':
            self.v = self.x
        elif data_to_analyze == 'y':
            self.v = self.y

    def import_lorentzians(self, p_table):
        """
        Import a 2D parameter array into the current Lorentzians array. This
        is somewhat deprecated and mainly used for calling `_Live_Instance`
        directly from a Jupyter notebook. Since this usecase is considered
        to be deprecated, this function is as well.

        Parameters
        ----------
        p_table : arr
            2D Lorentzian parameter array.
        """
        for i in range(0, len(p_table)):
            p = p_table[i]
            f0 = p[1]
            self.live_lorentzians[f0] = _Live_Lorentz(p)

    def activate(self, loop=True):
        """
        Set up the interface for the interactive tkinter window.

        Parameters
        ----------
        loop : bool, optional
            Deprecated. Determines whether the main tkinter loop will be
            running. This should always be true or else the window will stop
            updating and freeze. By default, True.
        """
        self.first_load = True
        self.fig = Figure()
        self.fig.set_size_inches(16, 9)
        self.ax = self.fig.add_subplot(111)
        self.root = tk.Tk(
            baseName='interactiveSession',
            className='liveSelector')
        self.root.wm_title("Live Peak Finder")
        self.controls = tk.Frame(self.root)
        self.plot_bar = tk.Frame(self.root)
        self.controls.pack(side="top", fill="both", expand=False)
        self.plot_bar.pack(side="top", fill="both", expand=False)
        self.slow_render = True
        if loop:
            self.make_button("Done", 
                command=self.close_window,
                description="Finish selecting and close the window <q>")
            self.make_button("Refresh",
                command=self.update_window,
                description="Force a refresh of the window")
            self.make_button("Add Lorentzians",
                command=self.add_lorentz,
                description="Add another Lorentzian with selected bounds <a>")
            self.make_button("Remove Lorentzians",
                command=self.remove_lorentz,
                description="Remove selected Lorentzian <d>")
            self.make_button("Show/Hide Components",
                command=self.components_bool,
                description="Toggle display of the Lorentzian x and y"
                " components <space>")
            self.make_button("Raise Components",
                command=self.raise_components,
                description="Raise the x and y components of the Lorentzians"
                " <↑>")
            self.make_button("Lower Components",
                command=self.lower_components,
                description="Lower the x and y components of the Lorentzians"
                " <↓>")
            self.make_button("Raise Projection",
                command=self.raise_projection,
                description="Raise the projection of the fitted Lorentzians"
                " <←>")
            self.make_button("Lower Projection",
                command=self.lower_projection,
                description="Lower the projection of the fitted Lorentzians"
                " <→>")
            self.make_button("Toggle Quick Render",
                command=self.toggle_quick_render,
                description="Enables \"quick render\" mode which updates"
                " faster but has some bugs <r>")
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.separator = ttk.Separator(self.root)
        self.separator.pack(in_=self.controls, side="left", padx=2)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_bar)
        self.toolbar.update()
        # self.canvas._tkcanvas.pack()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.plot_lorentzians()
        self.add_lorentz()
        if loop:
            tk.mainloop()

    def toggle_quick_render(self):
        """
        Enables "quick render" mode. This allows for faster updates in the
        matplotlib frame but also has more graphical bugs.
        """
        self.slow_render = not self.slow_render

    def make_button(self, text, command, description=None):
        """
        Make a tkinter button and add it to the GUI.

        Parameters
        ----------
        text : string
            Text to show in the button.
        command : function
            Method to run when the button is pressed.
        description : string
            Text for the tooltip. By default, None.
        """
        button = ttk.Button(master=self.root, text=text,
                            command=command, takefocus=False)
        button.pack(in_=self.controls, side="left")
        if description is not None:
            _Tooltip.create_tooltip(button, description)

    def close_window(self):
        """
        Safely disconnect canvas components and close the tkinter window.
        """
        self.root.quit()
        self.root.destroy()

    def update_window(self):
        """
        Update the current plots.
        """
        self.canvas.draw()

    def reset_axes(self):
        """
        Reset the axes to the default zoom level.
        """
        self.ax.set_xlim(self.default_x_lim)
        self.ax.set_ylim(self.default_y_lim)
        self.plot_lorentzians()

    def get_all_params(self):
        """
        Construct a 2D parameter array from the currently active Lorentzians.

        Returns
        -------
        arr
            2D Lorentzian parameter array.
        """
        p_table = np.empty((0, 4))
        for l in self.live_lorentzians:
            p = self.live_lorentzians[l].params()
            p_table = np.append(p_table, np.array([p]), axis=0)
        p_table = fl.correct_parameters(p_table)
        return p_table

    def plot_lorentzians(self):
        """
        Plot the currently saved Lorentzians.
        """
        if not self.first_load:
            x_lim = self.ax.get_xlim()
            y_lim = self.ax.get_ylim()
        if self.slow_render:
            self.ax.cla()
        p_table = self.get_all_params()
        if self.slow_render:
            if len(p_table) > 0:
                full_v = gl.multi_lorentz_2d(self.f, p_table)
            else:
                full_v = np.zeros((len(self.f),))
        if self.slow_render:
            offset = np.mean(self.v) + self.projection_height * \
                np.abs(min(self.v) - max(self.v))
        for i in range(0, len(p_table)):
            try:
                ex_f, ex_v = fl.lorentz_bounds_to_data(
                    p_table[i], self.f, self.v, expansion=2)
                self.ax.axvline(x=min(ex_f), color='pink')
                self.ax.axvline(x=max(ex_f), color='pink')
            except BaseException:
                pass
        if self.slow_render:
            self.ax.plot(self.f, self.v, color='b')
        if self.slow_render:
            self.ax.plot(self.f, full_v + offset, color='b')
        if self.slow_render:
            to_render = 0
        else:
            to_render = len(p_table) - 1
        for i in range(to_render, len(p_table)):
            try:
                og_f, og_v = fl.lorentz_bounds_to_data(
                    p_table[i], self.f, self.v, expansion=2)
                if self.slow_render:
                    ex_f, ex_v = fl.lorentz_bounds_to_data(
                        p_table[i], self.f, full_v, expansion=2)
                if self.all_data and self.show_components:
                    small_f, small_x = fl.lorentz_bounds_to_data(
                        p_table[i], self.f, self.x, expansion=2)
                    x_opt, x_cov = curve_fit(_lin, small_f, small_x)
                    x_fit = _lin(small_f, *x_opt)
                    self.ax.plot(small_f, small_x -
                                 x_fit +
                                 np.mean(og_v) +
                                 self.component_height *
                                 (np.max(og_v) -
                                  np.min(og_v)), color='y')
                    small_f, small_y = fl.lorentz_bounds_to_data(
                        p_table[i], self.f, self.y, expansion=2)
                    y_opt, y_cov = curve_fit(_lin, small_f, small_y)
                    y_fit = _lin(small_f, *y_opt)
                    self.ax.plot(small_f, small_y -
                                 y_fit +
                                 np.mean(og_v) +
                                 self.component_height *
                                 (np.max(og_v) -
                                  np.min(og_v)), color='g')
                self.ax.plot(og_f, og_v, color='r')
                if self.slow_render:
                    self.ax.plot(ex_f, ex_v + offset, color='r')
            except BaseException:
                pass
        if not self.first_load:
            self.ax.set_xlim(x_lim)
            self.ax.set_ylim(y_lim)
        else:
            self.default_x_lim = self.ax.get_xlim()
            self.default_y_lim = self.ax.get_ylim()
        self.first_load = False
        for vline in self.vlines:
            self.ax.axvline(vline, color='g', linestyle=':', linewidth=2)
        self.update_window()

    def set_vline(self, vline):
        """
        Stick a vertical green line at the specified point. This is used when
        calling a `_Live_Instance` from inside of an interactive color plot
        session. When selecting a point in the color plot for a "parameter
        preview" then the y coordinate is which set of frequency data to plot
        and the x coordinate is the frequency at which to draw the vertical
        green line.

        Parameters
        ----------
        vline : float
            The frequency at which to draw the vertical green line.
        """
        self.vlines.append(vline)

    def add_lorentz(self):
        """
        Tell the interface that it's time to add Lorentzians.
        """
        self.end_interactive()
        self.start_add_interactive()
        self.act_press = self.fig.canvas.mpl_connect(
            'key_press_event', self.on_press_add)

    def remove_lorentz(self):
        """
        Tell the interface that it's time to remove Lorentzians.
        """
        self.end_interactive()
        self.start_rem_interactive()
        self.act_press = self.fig.canvas.mpl_connect(
            'key_press_event', self.on_press_rem)

    def start_add_interactive(self):
        """
        Start the interactive session for "add more."
        """
        self.selection = None
        self.cursor = Cursor(self.ax, useblit=True,
                             color='0.5', linewidth=1, linestyle=":")
        self.act_select = RectangleSelector(
            self.ax,
            self.on_select,
            drawtype='box',
            useblit=False,
            button=[1, 3],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
            rectprops=dict(
                facecolor='grey',
                edgecolor='green',
                alpha=0.2,
                fill=True))

    def start_rem_interactive(self):
        """
        Start the interactive session for "remove mode."
        """
        self.selection = None
        self.cursor = Cursor(self.ax, useblit=True,
                             color='0.5', linewidth=1, linestyle=":")
        self.act_select = RectangleSelector(
            self.ax,
            self.on_select,
            drawtype='box',
            useblit=False,
            button=[1, 3],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
            rectprops=dict(
                facecolor='grey',
                edgecolor='red',
                alpha=0.2,
                fill=True))

    def end_interactive(self):
        """
        Pause the interactive session so that other processes can be run
        without causing bugs or so that the type of session can be switched.
        """
        try:
            self.canvas.mpl_disconnect(self.act_press)
        except BaseException:
            pass
        try:
            self.act_select.set_active(False)
        except BaseException:
            pass
        self.plot_lorentzians()

    def on_press_add(self, event):
        """
        Parse keystrokes and apply the correct keyboard shortcuts. There is a
        separate parser for "add mode" and "remove mode" in `_Live_Instance`.
        This is the parser ofr "add mode."

        Parameters
        ----------
        event : Matplotlib event.
            The event recorded from the user's keystroke.
        """
        if event.key == 'enter':
            min_ind = util.find_nearest_index(self.f, self.selection.f_min)
            max_ind = util.find_nearest_index(self.f, self.selection.f_max)
            region = np.array([[min_ind, max_ind]])
            region_f, region_v = util.extract_region(0, region, self.f, self.v)
            p_list = [
                fl.free_n_least_squares(
                    region_f,
                    region_v,
                    max_n=1,
                    force_fit=True,
                    method=self.method).x]
            p_table = fl.extract_parameters(p_list)
            self.import_lorentzians(p_table)
            self.end_interactive()
            self.plot_lorentzians()
            self.add_lorentz()
        elif event.key == 'escape':
            self.end_interactive()
        elif event.key == 'a':
            self.add_lorentz()
        elif event.key == 'd':
            self.remove_lorentz()
        elif event.key == 's':
            self.reset_axes()
        elif event.key == 'm':
            self.toolbar.pan()
        elif event.key == 'z':
            self.toolbar.zoom()
        elif event.key == 'h':
            self.toolbar.home()
        elif event.key == 'f':
            self.plot_lorentzians()
        elif event.key == 'q':
            self.close_window()
        elif event.key == 'r':
            self.toggle_quick_render()
        elif event.key == 'left':
            self.raise_projection()
        elif event.key == 'right':
            self.lower_projection()
        elif event.key == 'space':
            self.components_bool()
        elif event.key == 'up':
            self.raise_components()
        elif event.key == 'down':
            self.lower_components()

    def on_press_rem(self, event):
        """
        Parse keystrokes and apply the correct keyboard shortcuts. There is a
        separate parser for "add mode" and "remove mode" in `_Live_Instance`.
        This is the parser ofr "remove mode."

        Parameters
        ----------
        event : Matplotlib event.
            The event recorded from the user's keystroke.
        """
        if event.key == 'enter':
            to_remove = []
            for l in self.live_lorentzians:
                if l < self.selection.f_max and l > self.selection.f_min:
                    to_remove.append(l)
            for i in range(0, len(to_remove)):
                self.live_lorentzians.pop(to_remove[i])
            self.end_interactive()
            self.plot_lorentzians()
            self.remove_lorentz()
        elif event.key == 'escape':
            self.end_interactive()
        elif event.key == 'a':
            self.add_lorentz()
        elif event.key == 'd':
            self.remove_lorentz()
        elif event.key == 's':
            self.reset_axes()
        elif event.key == 'm':
            self.toolbar.pan()
        elif event.key == 'z':
            self.toolbar.zoom()
        elif event.key == 'h':
            self.toolbar.home()
        elif event.key == 'f':
            self.plot_lorentzians()
        elif event.key == 'q':
            self.close_window()
        elif event.key == 'r':
            self.toggle_quick_render()
        elif event.key == 'left':
            self.raise_projection()
        elif event.key == 'right':
            self.lower_projection()
        elif event.key == 'space':
            self.components_bool()
        elif event.key == 'up':
            self.raise_components()
        elif event.key == 'down':
            self.lower_components()

    def on_select(self, click, release):
        """
        Action to be done upon dragging out a selected window.

        Parameters
        ----------
        click : matplotlib event
            The event corresponding to the initial click.
        release : matplotlib event
            The event corresponding to when the click is released.
        """
        x1, y1 = click.xdata, click.ydata
        x2, y2 = release.xdata, release.ydata
        x_delta = np.abs(x1 - x2)
        y_delta = np.abs(y1 - y2)
        x_pos = min(x1, x2)
        y_pos = min(y1, y2)
        self.selection = _Selection(x_delta, y_delta, x_pos, y_pos)

    def components_bool(self):
        """
        Toggle whether or not the x and y Lorentzian components are shown.
        """
        self.show_components = not self.show_components
        self.plot_lorentzians()

    def raise_components(self):
        """
        Raise the x and y Lorentzian components by half of the projection's
        height.
        """
        self.component_height += 0.5
        self.plot_lorentzians()

    def lower_components(self):
        """
        Lower the x and y Lorenztian components by half of the projection's
        height.
        """
        self.component_height -= 0.5
        self.plot_lorentzians()

    def raise_projection(self):
        """
        Raise the projected Lorentzian fit by half of the projection's height.
        """
        self.projection_height += 0.5
        self.plot_lorentzians()

    def lower_projection(self):
        """
        Lowers the projected Lorentzian fit by half of the projection's height.
        """
        self.projection_height -= 0.5
        self.plot_lorentzians()


class _Color_Selector:
    """
    The class containing all the information for the interactive color plot
    session.
    """

    def __init__(
            self,
            data_files,
            x_res=1000,
            y_res=1,
            cmap='cool',
            parameters=None,
            method='lm',
            no_out=False):
        """
        Initial call the build the `_Color_Selector`.

        Parameters
        ----------
        data_files : list
            List of data files to process and display.
        x_res : int, optional
            Initial resolution of the displayed x-axis. Default is `1000` but
            can be set higher or lower depending on what sort of graphics your
            computer can handle. The maximum value is the length of the
            frequency array in each data file. The x-axis is the frequency
            axis.
        y_res : int, optional
            Initial resolution of the displayed y-axis. Default is `100` but
            can be set higher or lower depending on what sort of graphics your
            computer can handle. The maximum value is the length of the list of
            data files inputted into `data_files`. The y-axis just corresponds
            to the sweeps in the order that they're given in the inputted list
            of data files. For a linear change in temperature this means the
            values displayed are proportional to the temperature. But, they are
            not the temperature itself.
        cmap : str, optional
            Initial colormap used for the displayed plot. Defaults to `viridis`
            but accepts any colormap that would be usable by `matplotlib`. This
            can be changed from a selection of other colormaps during the
            interactive plotting process.
        parameters : arr, optional
            Any pre-determined Lorentzian parameters to continue working from.
            The parameters returned by `color_selection` work for this purpose.
            This exists so you can save your work during the fitting process
            and load it again later.
        method : {'trf', 'dogbox', 'lm'}, optional
            Fitting method used by the :func: `<scipy.optimize.least_squares>`
            backend for fitting to Lorentzians. See the documentation on that
            for more details. You generally do not have to change this.
        no_out : bool, optional
            Makes the default zoom slightly scaled down along the y-axis. This
            is to make it easier to use the rectangle zoom tool over a region
            without accidentially cutting out any data you need to fit over.
        """
        self.x_res = x_res
        self.y_res = y_res
        self.cmap = cmap
        self.data_files = data_files
        self.temps = util.get_temperatures(data_files)
        self.parameters = parameters
        self.method = method
        self.no_out = no_out
        self.setup_plot()
        self.setup_interface()
        self.setup_connections()
        self.setup_trackers()
        if parameters is not None:
            self.plot_parameters(parameters)
        tk.mainloop()

    def setup_interface(self):
        """
        Set up the tkinter GUI for interactive with the color plot session.
        """
        self.root = tk.Tk(
            baseName='interactiveSession',
            className='colorSelector')
        self.root.wm_title("Color Selector")
        self.patch_color = "tab:red"
        self.line_color = "tab:red"
        self.controls = tk.Frame(self.root)
        self.plot_bar = tk.Frame(self.root)
        self.controls.pack(side="top", fill="both", expand=False)
        self.plot_bar.pack(side="top", fill="both", expand=False)
        self.make_button("Done",
            command=self.close_window,
            description="Close the color selection window <q>")
        self.make_button("Another!", command=self.another_selection)
        self.make_button("Toggle Displays", command=self.toggle_show)
        self.make_button("Parameter Preview",
                         command=self.horizontal_selection)
        self.make_button("Point Info", command=self.what_temperature)
        self.make_button("Toggle Enhance!", command=self.enhance)
        self.make_button("(Un)Enhance!", command=self.unenhance)
        self.make_button("Inspire Me", command=self.inspire_me)
        self.make_button("Delete", command=self.enable_delete)
        self.make_button("Pre-Render", command=self.prerender)
        self.make_button("Tweak", command=self.tweak)
        self.make_button("Save", command=self.save)
        self.make_button("Axis", command=self.label_axes)
        self.make_button("Capture", command=self.capture)
        self.make_cmap_menu()
        self.make_color_menus()
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.separator = ttk.Separator(self.root, orient=tk.VERTICAL)
        self.separator.pack(in_=self.controls, side="left", padx=2)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_bar)
        self.toolbar.update()
        self.canvas._tkcanvas.pack()
        self.enhance_mode = False
        self.clean_mode = False
        self.path = None
        self.mode = 'select'  # Modes: select, enhance, preview, delete, move
        self.last_mode = None

    def setup_plot(self):
        self.ax, self.collection, self.colors = _make_colors(
            self.data_files,
            max_res=self.x_res,
            y_res=self.y_res,
            cmap=self.cmap)
        """
        Setup the color plot.

        See Also
        --------
        `_make_colors`
        """
        self.fig = self.ax.figure
        self.fig.set_size_inches(16, 9)
        self.xys = self.collection.get_offsets()
        self.Npts = len(self.xys)
        self.show_patches = True
        if not self.no_out:
            self.slight_zoom_out()

    def setup_trackers(self):
        """
        Define various trackers to hold the data for the interactive session.
        """
        self.ind = []
        self.selections = []
        self.patches = []
        self.lines = []
        self.selectors = []
        self.enhanced_renders = []
        self.enhanced_areas = []
        self.paths = []
        self.trace_coords = []
        self.traces = []
        self.toggle_delete = False
        self.smooth = False

    def setup_connections(self):
        """
        Establish all of the connections relating to the different modes that
        exist in the color plot session. This handles the definition of all
        visual selection cues so the user knows that they're doing and what
        mode they're in.
        """
        self.poly = PolygonSelector(self.ax, self.on_select, useblit=True)
        self.cursor = Cursor(self.ax, useblit=True,
                             color='blue', linewidth=1, linestyle=":")
        self.cursor.set_active(False)
        self.pre_cursor = Cursor(
            self.ax, useblit=True, color='black', linewidth=1, linestyle=":")
        self.pre_cursor.set_active(False)
        self.del_cursor = Cursor(
            self.ax, useblit=True, color='red', linewidth=1, linestyle=":")
        self.del_cursor.set_active(False)
        self.tra_cursor = Cursor(
            self.ax, useblit=True, color='white', linewidth=1, linestyle=":")
        self.tra_cursor.set_active(False)
        self.tra_cursor_alt = Cursor(
            self.ax, useblit=True, color='white', linewidth=1, linestyle='-.')
        self.tra_cursor_alt.set_active(False)
        self.rec_select = RectangleSelector(
            self.ax,
            self.on_rec_select,
            useblit=True,
            drawtype="box",
            rectprops=dict(
                facecolor='grey',
                edgecolor='blue',
                alpha=0.2,
                fill=True))
        self.rec_select.disconnect_events()
        self.alt_rec_select = RectangleSelector(
            self.ax,
            self.on_rec_delete,
            useblit=True,
            drawtype="box",
            rectprops=dict(
                facecolor='grey',
                edgecolor='red',
                alpha=0.2,
                fill=True))
        self.alt_rec_select.disconnect_events()
        self.lasso_select = LassoSelector(
            self.ax, self.make_curve, lineprops={'color': 'white'})
        self.lasso_select.disconnect_events()
        self.press = self.canvas.mpl_connect('key_press_event', self.on_press)

    def set_mode(self, mode):
        """
        Set which connections are active for the given mode and establish that
        mode as the current one.

        Parameters
        ----------
        mode : string
            The mode to initialize.
        """
        self.last_mode = self.mode
        self.autosave()
        self.disconnect_all()
        self.mode = mode
        if self.mode == 'select':
            self.poly.connect_default_events()
            self.poly.set_visible(True)
        elif self.mode == self.last_mode:
            self.set_mode('select')
        elif self.mode == 'enhance':
            self.cursor.set_active(True)
            self.rec_select.connect_default_events()
            self.rec_select.set_visible(True)
        elif self.mode == 'preview' or self.mode == 'temp':
            self.pre_cursor.set_active(True)
            self.pick = self.canvas.mpl_connect(
                'button_release_event', self.on_pre_click)
        elif self.mode == 'delete':
            self.del_cursor.set_active(True)
            self.alt_rec_select.connect_default_events()
            self.alt_rec_select.set_visible(True)
            self.pick = self.canvas.mpl_connect('pick_event', self.on_delete)
        elif self.mode == 'trace':
            self.tra_cursor.set_active(True)
            self.lasso_select.connect_default_events()
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def disconnect_all(self):
        """
        Turn off all connections. This is a reset switch so that a new mode can
        be set and the correct connections can be turned on.
        """
        self.cursor.set_active(False)
        self.del_cursor.set_active(False)
        self.pre_cursor.set_active(False)
        self.tra_cursor.set_active(False)
        self.poly.disconnect_events()
        self.rec_select.disconnect_events()
        self.alt_rec_select.disconnect_events()
        self.lasso_select.disconnect_events()
        self.canvas.mpl_disconnect('pick_event')
        self.canvas.mpl_disconnect('button_release_event')

    def enable_delete(self):
        """
        Toggle to turn on 'delete' mode.
        """
        self.set_mode('delete')

    def on_delete(self, event):
        """
        Method to interpret the interactions done during 'delete' mode.

        Parameters
        ----------
        event : Event
            What interaction occured.
        """
        if self.mode == 'delete':
            patch = event.artist
            try:
                i = self.patches.index(patch)
                source = 'patch'
            except BaseException:
                i = self.paths.index(patch)
                source = 'path'
            if source == 'patch':
                self.selections.pop(i)
                self.patches.pop(i)
            else:
                self.paths.pop(i)
                self.parameters = util.delete_parameters(self.parameters, i)
            patch.remove()
            self.canvas.mpl_disconnect('pick_event')
            self.del_cursor.set_active(False)
            self.canvas.draw_idle()
            self.canvas.flush_events()
            self.enable_delete()

    def make_cmap_menu(self):
        """
        Make the menu of background color map options to use.
        """
        options = sorted(_option_colors.keys())
        self.color_variable = tk.StringVar(self.root)
        self.opt = ttk.OptionMenu(self.root, self.color_variable, *options)
        self.opt.pack(in_=self.controls, side="right")
        self.color_variable.set("Color Map")
        self.color_variable.trace("w", self.update_cmap)

    def make_color_menus(self):
        """
        Make the menu of colors to use for rendered curves on the plot.
        """
        options = sorted(_option_select_colors.keys())
        self.face_color_variable = tk.StringVar(self.root)
        self.param_color_variable = tk.StringVar(self.root)
        self.face_opt = ttk.OptionMenu(
            self.root, self.face_color_variable, *options)
        self.param_opt = ttk.OptionMenu(
            self.root, self.param_color_variable, *options)
        self.face_opt.pack(in_=self.controls, side="right")
        self.param_opt.pack(in_=self.controls, side="right")
        self.face_color_variable.set("Selection Color")
        self.param_color_variable.set("Fit Color")
        self.face_color_variable.trace("w", self.update_patch_color)
        self.param_color_variable.trace("w", self.update_line_color)

    def update_patch_color(self, *args):
        """
        Change the colors of all the drawn polygonal patches.
        """
        patch_color = _option_select_colors[self.face_color_variable.get()]
        for patch in self.patches:
            patch.set_facecolor(patch_color)
        self.patch_color = patch_color
        self.canvas.draw_idle()

    def update_line_color(self, *args):
        """
        Change the colors of all the drawn curves.
        """
        line_color = _option_select_colors[self.param_color_variable.get()]
        for line in self.lines:
            line.set_color(line_color)
        for path in self.paths:
            path.set_color(line_color)
        for trace in self.traces:
            trace.set_color(line_color)
        self.line_color = line_color
        self.canvas.draw_idle()

    def update_cmap(self, *args):
        """
        Change the background color map.

        See Also
        --------
        `update_color_map`
        """
        self.cmap = _option_colors[self.color_variable.get()]
        self.update_colors(cmap=self.cmap)

    def reload(self):
        """
        Reload the current color map and all plots on it. This completely
        refreshes the canvas.
        """
        self.autosave()
        self.canvas._tkcanvas.pack_forget()
        ax, collection = _make_colors(
            self.data_files, max_res=self.x_res, cmap=self.cmap)
        self.fig = ax.figure
        self.collection = collection
        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        self.ax = ax
        self.another_selection()
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="bottom", expand=True)
        self.canvas._tkcanvas.pack()

    def on_select(self, verts):
        """
        Actions to take upon a polygonal selction.

        Parameters
        ----------
        verts : array-like
            The transparent points within the polygon selction.

        See Also
        --------
        `_make_colors`
        """
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.canvas.draw_idle()

    def on_press(self, event):
        """
        Method that determines what's done for most of the possible keyboard
        shortcuts. The only exceptions are those that are handled automatically
        by matplotlib for rectangular selections, polygon selections, etc.

        Parameters
        ----------
        event : Event
            The completed interaction to be processed.
        """
        if event.key == 'enter':
            self.another_selection()
        elif event.key == ' ':
            self.toggle_show()
        elif event.key == 'e':
            self.enhance()
        elif event.key == 'x':
            self.unenhance()
        elif event.key == 'p':
            self.horizontal_selection()
        elif event.key == 'i':
            self.inspire_me()
        elif event.key == 'd':
            self.enable_delete()
        elif event.key == 'r':
            self.prerender()
        elif event.key == 'y':
            self.show_stuff()
        elif event.key == 'f':
            self.canvas.draw_idle()
            self.canvas.flush_events()
            self.canvas.draw_idle()
        elif event.key == 's':
            self.save()
        elif event.key == 'm':
            self.toolbar.pan()
        elif event.key == 'z':
            self.toolbar.zoom()
        elif event.key == 'h':
            self.toolbar.home()
        elif event.key == 'q':
            self.close_window()
        elif event.key == 't':
            self.tweak()
        elif event.key == 'a':
            self.trace()
        elif event.key == 'c':
            self.commit_trace()
        elif event.key == 'o':
            self.toggle_smooth()
        elif event.key == 'k':
            self.what_temperature()
        elif event.key == 'l':
            self.label_axes()
        elif event.key == 'u':
            self.capture()

    def disconnect(self):
        """
        Disconnect the polygon events and connections.
        """
        self.autosave()
        self.poly.disconnect_events()
        self.canvas.draw_idle()

    def toggle_smooth(self):
        """
        Toggles drawing of smooth curves when creating curves manually instead
        of fitting.
        """
        self.smooth = not self.smooth
        self.tra_cursor.set_active(False)
        self.tra_cursor, self.tra_cursor_alt = self.tra_cursor_alt, self.tra_cursor
        if self.mode == 'trace':
            self.tra_cursor.set_active(True)
        self.canvas.draw_idle()

    def capture(self):
        """
        Capture a high resolution snapshot of the current figure.
        """
        filters = (
            ("portable network graphics", "*.png"),
            ("portable document format", "*.pdf"),
            ("scalable vector graphics", "*.svg"),
            ("all files", "*.*"))
        name = 'Color Plot'
        fp = util._save_file(filters=filters, name=name)
        dpi = simpledialog.askinteger("DPI", "How many dots per inch?")
        self.fig.savefig(fp, dpi=dpi)
        self.fig.set_dpi(72)
        self.canvas.draw_idle()

    def tweak(self):
        """
        Make an interactive `mistake_selection` session for tweaking the
        current parameters.

        See Also
        --------
        `mistake_selection`
        """
        self.autosave()
        try:
            if self.parameters is not None and len(self.parameters[0]) > 0:
                self.parameters = mistake_selection(
                    self.data_files, self.parameters, self.path)
            else:
                tk.messagebox.showinfo('Error',
                                       'You have no parameters to tweak.')
        except BaseException:
            tk.messagebox.showinfo('Error', 'You have no parameters to tweak.')
        self.refresh_parameters()
        self.autosave()
        self.canvas.draw_idle()

    def trace(self):
        """
        Enable 'trace' mode for manually drawing traces instead of fitting.
        """
        self.canvas.draw_idle()
        self.set_mode('trace')

    def commit_trace(self):
        """
        Save the current trace.
        """
        trace = self.trace_coords
        p = self.parameters_from_trace(trace)
        for t in self.traces:
            t.set_alpha(0)
            t.remove()
        self.traces = []
        self.trace_coords = []
        if self.parameters is None:
            self.parameters = np.array(p)
        else:
            self.parameters = np.append(self.parameters, p, axis=1)
        self.refresh_parameters()

    def parameters_from_trace(self, trace):
        """
        Create parameters from the provided trace. Since this wasn't created
        by fitting, the only physically meaningful parameters is the frequency.
        But, the other parameters can't be listed as `None` since those will
        be filtered out by other methods. So, they're instead provided with
        approximate dummy values.

        Parameters
        ----------
        trace : array-like
            Vertices of the values from the trace.

        Returns
        -------
        arr
            A 3D Lorentzian parameter array containing all the parameters
            approximated by the trace.
        """
        p = []
        y = self.to_y(trace)
        y_min = min(y)
        old_scale = (0, self.x_res, self.x_res)
        new_scale = (min(self.data_files[0].f), max(
            self.data_files[0].f), len(self.data_files[0].f))
        for i in range(len(self.data_files)):
            if i in y:
                f0 = cd.normalize_0d(trace[i - y_min][0], old_scale, new_scale)
                i0 = util.find_nearest_index(self.data_files[0].f, f0)
                iin = max(0, i0 - 3)
                ifi = min(len(self.data_files[0].f) - 1, i0 + 3)
                fin = self.data_files[0].f[iin]
                ffi = self.data_files[0].f[ifi]
                FWHM = np.abs(ffi - fin)
                A = np.abs(max(self.data_files[0].r[iin:ifi + 1]) -
                           min(self.data_files[0].r[iin:ifi + 1]))
                p.append([[A, f0, FWHM, np.pi]])
            else:
                p.append([[np.nan, np.nan, np.nan, np.nan]])
        print(np.array(p))
        return np.array(p)

    def make_curve(self, verts):
        """
        Draw a curve from the provided vertices.

        Parameters
        ----------
        verts : array-like
            Vertices of the curve.
        """
        v = self.make_safe(verts)
        for trace in self.traces:
            trace.set_alpha(0)
        if self.smooth:
            v_backup = v
            try:
                x = self.to_x(v)
                y = self.to_y(v)
                xhat = savgol_filter(x, 15, 3)
                v = self.to_v(xhat, y)
            except BaseException:
                v = v_backup
        self.trace_coords += self.make_safe(v)
        self.trace_coords = self.make_safe(self.trace_coords)
        x = self.to_x(self.trace_coords)
        y = self.to_y(self.trace_coords)
        if len(x) == len(y) and len(x) > 0:
            trace_faint = self.ax.plot(
                x, y, color=self.line_color, alpha=0.5)[0]
            trace_strong = self.ax.plot(
                x, y, color=self.line_color, linestyle=':')[0]
            self.traces.append(trace_faint)
            self.traces.append(trace_strong)
            self.canvas.draw_idle()

    def to_v(self, x, y):
        """
        Create a vertex from the given coordinates.

        Parameters
        ----------
        x : arr
            1D array of the x value indices along the plot coordinates.
        y : int
            1D array of the y value indices along the plot coordinates.

        Returns
        -------
        list
            The "array-like" list of vertices.
        """
        v = []
        for i in range(len(x)):
            v.append((x[i], y[i]))
        return v

    def make_safe(self, verts):
        """
        Given a list of vertices, provides a new one where all the vertices
        are within the confines of the plot.

        Parameters
        ----------
        verts : list
            The list of input vertices.

        Returns
        -------
        list
            The list of output vertices.
        """
        x_vals = []
        y_vals = []
        verts.reverse()
        for v in verts:
            if not any([int(np.round(v[1])) in y_vals, v[1] < 0, v[1] >= len(
                    self.data_files), v[0] < 0, v[0] >= self.x_res]):
                x_vals.append(v[0])
                y_vals.append(int(np.round(v[1])))
        new_verts = []
        for i in range(len(x_vals)):
            new_verts.append((x_vals[i], y_vals[i]))
        new_verts.sort(key=lambda v: v[1])
        x = self.to_x(new_verts)
        y = self.to_y(new_verts)
        f = interpolate.interp1d(y, x)
        y_new = np.arange(min(y), max(y) + 1, 1)
        x_new = f(y_new)
        final_verts = []
        for i in range(len(y_new)):
            final_verts.append((x_new[i], int(np.round(y_new[i]))))
        return final_verts

    def to_x(self, verts):
        """
        Get the x indices corresponding to the given vertices.

        Parameters
        ----------
        verts : list
            List of vertices.

        Returns
        -------
        array
            The x values.
        """
        x = [v[0] for v in verts]
        return np.array(x)

    def to_y(self, verts):
        """
        Get the y indices corresponding to the given vertices.

        Parameters
        ----------
        verts : list
            List of vertices.

        Returns
        -------
        array
            The y values.
        """
        y = [v[1] for v in verts]
        return np.array(y)

    def refresh_parameters(self):
        """
        Clean up the parameters maintained by the session and re-render them
        correctly.
        """
        self.autosave()
        for path in self.paths:
            try:
                path.remove()
            except BaseException:
                pass
        self.parameters = fl.denan_parameters(self.parameters, 0.99)
        self.plot_parameters(self.parameters)
        self.canvas.draw_idle()

    def on_rec_delete(self, click, release):
        """
        What to do when a rectangular selection is drawn in 'delete' mode.

        Parameters
        ----------
        click : click event
            Data corresponding to the click that started drawing the rectangle.
        release : release event
            Data corresponding to the release that ended drawing the rectangle.
        """
        if self.mode == 'delete':
            x1, y1 = click.xdata, click.ydata
            x2, y2 = release.xdata, release.ydata
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            full_res = len(self.data_files[0].f)
            y_min = int(np.round(y_min))
            y_max = int(np.round(y_max))
            x_min = int(np.round(x_min / self.x_res * full_res))
            x_max = int(np.round(x_max / self.x_res * full_res))
            f_regions = []
            for i in range(len(self.data_files)):
                if i < y_min or i > y_max:
                    f_regions.append([np.inf, -np.inf])
                else:
                    if x_min < 0:
                        f_min = -np.inf
                    elif x_min > full_res:
                        f_min = self.data_files[i].f[-1]
                    else:
                        f_min = self.data_files[i].f[x_min]
                    if x_max > full_res:
                        f_max = np.inf
                    elif x_max < 0:
                        f_max = self.data_files[i].f[0]
                    else:
                        f_max = self.data_files[i].f[x_max]
                    f_regions.append([f_min, f_max])
            f_regions = np.array(f_regions)
            self.parameters = util.delete_parameters_from_f_regions_3d(
                self.parameters, f_regions)
            self.refresh_parameters()

    def on_rec_select(self, click, release):
        """
        What to do when a rectangular selection is drawn in 'enhance' mode.

        Parameters
        ----------
        click : click event
            Data corresponding to the click that started drawing the rectangle.
        release : release event
            Data corresponding to the release that ended drawing the rectangle.
        """
        if self.mode == 'enhance':
            try:
                self.cursor.set_active(True)
                self.canvas.draw_idle()
                self.canvas.flush_events()
                x1, y1 = click.xdata, click.ydata
                x2, y2 = release.xdata, release.ydata
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                self.render_enhance(x_min, x_max, y_min, y_max)
            except BaseException:
                pass

    def make_button(self, text, command, description=None):
        """
        Make a tkinter button and add it to the GUI.

        Parameters
        ----------
        text : string
            Text to show in the button.
        command : function
            Method to run when the button is pressed.
        description : string
            Text for the tooltip. By default None.
        """
        button = ttk.Button(master=self.root, text=text,
                            command=command, takefocus=False)
        button.pack(in_=self.controls, side="left")
        if description is not None:
            _Tooltip.create_tooltip(button, description)

    def close_window(self):
        """
        Disconnect everything and end the session when the window is closed.
        """
        try:
            self.another_selection()
        except BaseException:
            pass
        self.canvas._tkcanvas.pack_forget()
        self.toolbar.pack_forget()
        self.disconnect()
        self.canvas.mpl_disconnect(self.press)
        self.root.quit()
        self.root.destroy()

    def finish_selection(self):
        """
        Commit, render, and cleanup a polygonal selection.
        """
        selection = _points_to_ranges(
            np.array(self.xys[self.ind]), self.data_files, self.x_res)
        self.selections.append(selection)
        patch = Polygon(
            self.poly.verts,
            facecolor=self.patch_color,
            alpha=0.3,
            edgecolor='black',
            linestyle=':',
            linewidth=2,
            picker=True)
        self.patches.append(patch)
        self.disconnect()
        self.selectors.append(self.poly)
        self.ax.add_patch(patch)

    def another_selection(self):
        """
        Start another polygonal selection.
        """
        self.finish_selection()
        self.poly = PolygonSelector(self.ax, self.on_select, useblit=True)
        self.set_mode('select')
        self.ind = []

    def toggle_show(self):
        """
        Toggle the display of polygonal patches (selections), lines/curves,
        and the Lorentzian markers for individual temperatures.
        """
        self.autosave()
        self.show_patches = not self.show_patches
        if self.show_patches:
            for patch in self.patches:
                patch.set_alpha(0)
            for selector in self.selectors:
                selector.set_visible(False)
            for line in self.lines:
                line.set_alpha(0)
            for path in self.paths:
                path.set_alpha(0)
        else:
            for patch in self.patches:
                patch.set_alpha(0.3)
            for line in self.lines:
                line.set_alpha(1)
            for path in self.paths:
                path.set_alpha(1)
        self.canvas.draw_idle()

    def slight_zoom_out(self):
        """
        Slightly zoom out from the initial window so that zooming in can be
        done that includes the data from all possible temperatures. Without
        this, the data will exactly fill up the plot window so that any
        imperfect zoom will inevitably make some data go off the screen.
        """
        y_lim = self.ax.get_ylim()
        y_delta = y_lim[1] - y_lim[0]
        new_y_lim = (y_lim[0] - 0.1 * y_delta, y_lim[1] + 0.1 * y_delta)
        self.ax.set_ylim(new_y_lim[0], new_y_lim[1])

    def make_params_from_selection(self, index):
        """
        Make parameters from the selection corresponding to the provided index.

        Parameters
        ----------
        index : int
            Index of the `Data_File` to make the parameters from.

        Returns
        -------
        arr
            A 3D Lorentzian parameter array.
        """
        params = np.empty((0, 4))
        for i in range(0, len(self.selections)):
            try:
                f = self.data_files[index].f
                v = self.data_files[index].r
                regions = self.selections[i][index]
                p = fl.parameters_from_regions(
                    f,
                    v,
                    regions,
                    catch_degeneracies=False,
                    method=self.method)
                params = np.append(params, p, axis=0)
            except BaseException:
                pass
        return params

    def label_axes(self):
        """
        Apply physically meaningful labels to the axes of the color plot.

        Notes
        -----
        This feature is still somewhat experimental. It hasn't been tested with
        different directions of temperature changing nor in cases where the
        material warms up and then cools down or vice versa.
        """
        x_ticks = simpledialog.askinteger(
            "Tick Preference", "How many ticks on the x-axis?")
        y_ticks = simpledialog.askinteger(
            "Tick Preference", "How many ticks on the y-axis?")
        x = np.arange(self.x_res)
        y = np.arange(len(self.data_files))
        f_selected = np.linspace(
            self.data_files[0].f[0], self.data_files[0].f[-1], self.x_res)
        f_unit = ' (Hz)'
        if min(f_selected) > 1000:
            f_selected /= 1000
            f_unit = ' (kHz)'
        if min(f_selected) > 1000:
            f_selected /= 1000
            f_unit = ' (MHz)'
        if min(f_selected) > 1000:
            f_selected /= 1000
            f_unit = ' (GHz)'
        x = cd.normalize_1d(x, (x[0], x[-1], x_ticks))
        y = cd.normalize_1d(y, (y[0], y[-1], y_ticks))
        x_labels = cd.normalize_1d(
            f_selected, (f_selected[0], f_selected[-1], x_ticks))
        y_labels = cd.normalize_1d(
            self.temps, (self.temps[0], self.temps[-1], y_ticks))
        f_digits = 0
        u = np.unique(np.round(x_labels, f_digits))
        while len(u) < x_ticks and np.round(
                min(x_labels), f_digits) != min(x_labels):
            f_digits += 1
            u = np.unique(np.round(x_labels, f_digits))
        T_digits = 0
        u = np.unique(np.round(y_labels, T_digits))
        while len(u) < y_ticks and np.round(
                min(y_labels), T_digits) != min(y_labels):
            T_digits += 1
            u = np.unique(np.round(y_labels, T_digits))
        x_labels = [str(np.round(f, f_digits)) for f in x_labels]
        y_labels = [str(np.round(T, T_digits)) for T in y_labels]
        # y_labels = np.flip(np.array(y_labels))
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(x_labels)
        self.ax.set_yticks(y)
        self.ax.set_yticklabels(y_labels)
        self.ax.set_xlabel('Frequency' + f_unit)
        self.ax.set_ylabel('Temperature (K)')
        self.canvas.draw_idle()

    def find_params(self, index, x):
        """
        Start a `live_selection` interactive session in a new window to find
        peaks so that you know where to look in the color plot.

        Parameters
        ----------
        index : int
            Index of the `Data_File` to look at.
        x : float
            The x value of the selected location so that a green reference line
            can be rendered in the `live_selection` session.

        Returns
        -------
        arr
            A 2D Lorentzian parameter array.

        See Also
        --------
        `live_selection`
        """
        existing_params = self.make_params_from_selection(index)
        if self.parameters is not None:
            for i in range(0, len(self.parameters[index])):
                p = self.parameters[index][i]
                if not any(np.isnan(p)):
                    existing_params = np.append(existing_params, [p], axis=0)
        params = live_selection(
            self.data_files[index], params=existing_params, vline=x)
        return params

    def display_params(self, params, index):
        """
        Display the given parameters at the `Data_File` corresponding to the
        provided index.

        Parameters
        ----------
        params : arr
            A 2D Lorentzian parameter array.
        index : int
            Index of the `Data_File` to display show the Lorentzian markers at.
        """
        f = self.data_files[index].f
        regions = fl.regions_from_parameters(f, params, extension=2)
        y = np.array([index, index])
        for i in range(0, len(regions)):
            x = regions[i] * self.x_res / len(f)
            line = Line2D(x, y, color=self.line_color)
            self.lines.append(line)
            self.ax.add_line(line)
        self.canvas.draw_idle()

    def horizontal_selection(self):
        """
        Enable 'preview' mode.
        """
        self.set_mode('preview')

    def what_temperature(self):
        """
        Enable 'temp' mode.
        """
        self.set_mode('temp')

    def on_pre_click(self, event):
        """
        Actions correspondong to events where the user clicks on the color plot
        with cross-hairs active. This is what should be done in both 'preview'
        and 'temp' modes.

        Parameters
        ----------
        event : click event
            The click done at the targeted location on the plot.
        """
        x_select = event.xdata
        y_select = event.ydata
        if x_select is None:
            x_select = -1
        if y_select is None:
            y_select = -1
        x_check = x_select > 0 and x_select < self.x_res
        index = int(np.round(y_select))
        y_check = index >= 0 and index < len(self.data_files)
        if self.mode == 'preview':
            if x_check and y_check:
                f = self.data_files[index].f
                x = (event.xdata / self.x_res) * (f[-1] - f[0]) + f[0]
                self.canvas.mpl_disconnect('button_release_event')
                self.set_mode('select')
                params = self.find_params(index, x)
                self.display_params(params, index)
        elif self.mode == 'temp':
            if x_check and y_check:
                f = self.data_files[index].f
                f0 = (event.xdata / self.x_res) * (f[-1] - f[0]) + f[0]
                f0 = np.round(f0, 2)
                f_string = str(f0) + ' Hz'
                if f0 > 1000:
                    f0 = np.round(f0 / 1000, 2)
                    f_string = str(f0) + ' kHz'
                if f0 > 1000:
                    f0 = np.round(f0 / 1000, 2)
                    f_string = str(f0) + ' MHz'
                temp = self.temps[index]
                show_string = ('Index: ' + str(index) + '\nTemperature: ' +
                               str(temp) + ' K\nFrequency: ' + f_string)
                tk.messagebox.showinfo('Point Info', show_string)

    def inspire_me(self):
        """
        Use machine learning to try and find some Lorentzians automatically.
        """
        if _can_ml:
            self.autosave()
            index = simpledialog.askinteger("Index Selection", "Which index?")
            params = auto.quick_analyze(
                self.data_files[index].f, self.data_files[index].r)
            self.display_params(params, index)
        else:
            auto.quick_analyze(None, None)

    def update_colors(self, cmap="viridis"):
        """
        Render an updated backgroudn color map.

        Parameters
        ----------
        cmap : str, optional
            The new color map, by default "viridis."

        See Also
        --------
        `update_cmap`
        """
        max_res = self.x_res
        y_res = self.y_res
        data_files = self.data_files
        z = np.empty((0, max_res))
        y_skip = int(np.round(1 / y_res))
        for i in range(0, len(data_files)):
            if i % y_skip == 0:
                z0 = data_files[i].r
                z0 = cd.scale_zoom(z0, 0, 1)
                z0 = cd.normalize_1d(z0, (min(z0), max(z0), max_res))
                coef = np.polyfit(np.arange(max_res), z0, 3)
                bg = np.poly1d(coef)
                z0 -= bg(np.arange(0, max_res))
                z = np.append(z, [z0], axis=0)
        color_x = np.linspace(0, max_res, max_res)
        color_y = np.linspace(0, len(data_files), len(z))
        self.colors.remove()
        self.colors = self.ax.pcolormesh(
            color_x, color_y, z, cmap=cmap, shading='auto')
        for render in self.enhanced_renders:
            render.remove()
        self.enhanced_renders = []
        areas_to_enhance = self.enhanced_areas
        self.enhanced_areas = []
        for area in areas_to_enhance:
            self.render_enhance(area[0], area[1], area[2], area[3])
        self.canvas.draw_idle()

    def enhance(self):
        """
        Activate 'enhance' mode.
        """
        self.autosave()
        self.set_mode('enhance')

    def unenhance(self):
        """
        Turn off all enhancements for faster rendering.
        """
        for render in self.enhanced_renders:
            render.remove()
        self.enhanced_renders = []
        self.enhanced_areas = []
        self.canvas.draw_idle()

    def renormalize(self, verts):
        """
        Normalized vertices for an enhanced region.

        Parameters
        ----------
        verts : array-like
            Input vertices.

        Returns
        -------
        list
            Normalized vertices.
        """
        full_min = 0
        full_max = len(self.data_files[0].f) - 1
        x_min = 0
        x_max = self.x_res
        normed_verts = []
        old_scale = (x_min, x_max, x_max)
        new_scale = (full_min, full_max, full_max)
        for v in verts:
            x = cd.normalize_0d(v[0], old_scale, new_scale)
            y = v[1]
            normed_verts.append((x, y))
        return normed_verts

    def render_enhance(self, x_min, x_max, y_min, y_max):
        """
        Render an enhanced area.

        Parameters
        ----------
        x_min : float
            Minimum x value of enhanced area.
        x_max : float
            Maximum x value of enhanced area.
        y_min : float
            Minimum y value of enhanced area.
        y_max : float
            Maximum y value of enhanced area.
        """
        x_min = max(x_min, 0)
        x_max = min(x_max, self.x_res)
        y_min = max(y_min, 0)
        y_max = min(y_max, len(self.data_files))
        full_res = len(self.data_files[0].f)
        y_min = int(np.round(y_min))
        y_max = int(np.round(y_max))
        full_x_min = int(np.round(x_min / self.x_res * full_res))
        full_x_max = int(np.round(x_max / self.x_res * full_res))
        max_res = len(self.data_files[0].f[full_x_min:full_x_max])
        data_files = self.data_files
        z = np.empty((0, max_res))
        for i in range(y_min, y_max):
            z0 = data_files[i].r[full_x_min:full_x_max]
            z0 = cd.scale_zoom(z0, 0, 1)
            z0 = cd.normalize_1d(z0, (min(z0), max(z0), max_res))
            coef = np.polyfit(np.arange(max_res), z0, 3)
            bg = np.poly1d(coef)
            z0 -= bg(np.arange(0, max_res))
            z = np.append(z, [z0], axis=0)
        color_x = np.linspace(x_min, x_max, max_res)
        color_y = np.linspace(y_min, y_max, len(z))
        self.enhanced_renders.append(self.ax.pcolormesh(
            color_x, color_y, z, cmap=self.cmap, shading='auto'))
        new_area = (x_min, x_max, y_min, y_max)
        bad_areas = []
        for i in range(0, len(self.enhanced_areas)):
            old_area = self.enhanced_areas[i]
            if _check_contains(old_area, new_area):
                bad_areas.append(i)
                self.enhanced_renders[i].remove()
        for i in range(len(bad_areas) - 1, -1, -1):
            self.enhanced_renders.pop(bad_areas[i])
            self.enhanced_areas.pop(bad_areas[i])
        self.enhanced_areas.append(new_area)
        self.canvas.draw_idle()

    def plot_parameters(self, parameters):
        """
        Plot the provided parameters.

        Parameters
        ----------
        parameters : arr
            A 3D Lorentzian parameter array.
        """
        for i in range(0, len(parameters[0])):
            x = self.x_res * (parameters[..., i, 1] - min(self.data_files[0].f)) / (
                max(self.data_files[0].f) - min(self.data_files[0].f))
            y = np.arange(len(parameters))
            self.paths.append(self.ax.plot(
                x, y, color=self.line_color, picker=False)[0])

    def prerender(self):
        """
        Render fit selections without leaving the interactive session.
        """
        self.autosave()
        self.parameters = fl.parameters_from_selections(
            self.data_files, (self.selections, self.parameters),
            method=self.method)
        self.refresh_parameters()
        self.selections = []
        for patch in self.patches:
            patch.remove()
        self.patches = []
        self.canvas.draw_idle()

    def show_stuff(self):
        """
        Basic debugging information.
        """
        print()
        print('----- debug -----')
        print()
        print('selections:')
        print(len(self.selections))
        print()
        print('parameters:')
        print(self.parameters.shape)
        print()
        print('paths')
        print(len(self.paths))

    def save(self):
        """
        Open a save window. Saves a pickled tuple of the selections and
        parameters.
        """
        if self.path is None:
            name = 'selections'
        else:
            name = self.path.split('/')[-1]
        self.path = util.save((self.selections, self.parameters), name=name)

    def autosave(self):
        """
        Once one save is done, autosave is enabled automatically.
        """
        if self.path is not None:
            path = self.path[:-4] + '_autosave' + '.pkl'
            util.save((self.selections, self.parameters), path)


class _Point_Selector:
    """
    The class containing all the information for the interactive point picking
    and removing session.

    See Also
    --------
    `point_selection`
    """

    def __init__(self, data_files, fs=[]):
        self.data_files = data_files
        self.picked = []
        self.pick = True
        self.chosen_frequencies = fs
        self.setup_interface()

    def setup_interface(self):
        """
        Set up the tkinter GUI for interactive with the color plot session.
        """
        self.root = tk.Tk(
            baseName='interactiveSession',
            className='pointSelector')
        self.root.wm_title("Frequency Selector")
        self.controls = tk.Frame(self.root)
        self.controls.pack(side="top", fill="both", expand=False)
        self.make_button("Done",
            command=self.close_window,
            description="Finish selecting and close window")
        self.make_button("Delete",
            command=self.toggle_delete,
            description="Remove duplicate points")
        self.fig = Figure()
        self.fig.set_size_inches(16, 9)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill='both', expand=True)
        self.separator = ttk.Separator(self.root)
        self.separator.pack(in_=self.controls, side="left", padx=2)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.controls)
        self.toolbar.update()
        self.canvas._tkcanvas.pack()
        self.plot_points()
        self.cursor = Cursor(self.ax, useblit=True,
                             color='black', linewidth=1, linestyle=":")
        if len(self.chosen_frequencies) > 0:
            for f0 in self.chosen_frequencies:
                self.picked.append(
                    (f0, self.ax.axvline(f0, color='black', linewidth=1)))
        tk.mainloop()

    def make_button(self, text, command, description=None):
        """
        Make a tkinter button and add it to the GUI.

        Parameters
        ----------
        text : string
            Text to show in the button.
        command : function
            Method to run when the button is pressed.
        description : string
            Text for the tooltip. By default None.
        """
        button = ttk.Button(master=self.root, text=text,
                            command=command, takefocus=False)
        button.pack(in_=self.controls, side="left")
        if description is not None:
            _Tooltip.create_tooltip(button, description)

    def close_window(self):
        """
        Safely disconnect canvas components and close the tkinter window.
        """
        self.canvas._tkcanvas.pack_forget()
        self.toolbar.pack_forget()
        self.canvas.mpl_disconnect(self.on_pick)
        self.root.quit()
        self.root.destroy()

    def toggle_delete(self):
        """
        Switch to the "delete mode" cursor.
        """
        self.cursor.set_active(False)
        self.pick = not self.pick
        if self.pick:
            self.cursor = Cursor(self.ax, useblit=True,
                                 color='black', linewidth=1, linestyle=":")
        else:
            self.cursor = Cursor(self.ax, useblit=True,
                                 color='red', linewidth=1, linestyle=":")

    def plot_points(self):
        """
        Plot the current points.
        """
        for i in range(len(self.data_files)):
            scale = (0, 1, len(self.data_files[i].r))
            f = self.data_files[i].f
            v = cd.normalize_1d(self.data_files[i].r, scale)
            just_plotted = self.ax.plot(f, v + i, alpha=0.5)
            color = just_plotted[0].get_color()
            if not self.data_files[i].params is None:
                pts_f = self.data_files[i].params[..., 1]
                pts_v = util._scatter_pts(pts_f, f, v)
                self.ax.scatter(pts_f, pts_v + i, color=color, picker=5)
            self.ax.text(f[0], v[0] +
                         i, str(self.data_files[i].T[0]) +
                         ' K ', color=color, ha='right')
            self.ax.text(f[-1],
                         v[-1] + i,
                         ' ' + str(self.data_files[i].T[-1]) + ' K',
                         color=color,
                         ha='left')
        self.canvas.mpl_connect('pick_event', self.on_pick)

    def on_pick(self, event):
        """
        Trigger for actions to be done upon picking a point or just clicking in
        general.

        Parameters
        ----------
        event : matplotlib event
            The click.
        """
        coords = event.artist.get_offsets()[event.ind][0]
        color = event.artist.get_facecolors()[0].tolist()
        f0 = coords[0]
        v0 = coords[1]
        if self.pick:
            self.chosen_frequencies.append(f0)
            self.ax.scatter([f0], [v0], color='black')
            self.picked.append(
                (f0, self.ax.axvline(f0, color='black', linewidth=1)))
        else:
            ind_1 = util.find_nearest_index(
                np.array(self.chosen_frequencies), f0)
            f = []
            for i in range(0, len(self.picked)):
                f.append(self.picked[i][0])
            ind_2 = util.find_nearest_index(np.array(f), f0)
            self.picked[ind_2][1].set_alpha(0)
            self.picked.pop(ind_2)
            self.ax.scatter([f0], [v0], color=color)
            self.toggle_delete()
        self.canvas.draw_idle()


class _Mistake_Selector():
    """
    Backend for the `mistake_selection` function. This also powers the "tweak"
    used in interactive color plot sessions.
    """

    def __init__(self, data_files, parameters=None, path=None, method='lm'):
        """
        Create an instance of the `_Mistake_Selector` class.

        Parameters
        ----------
        data_files : list
            List of `Data_Files` to be used.
        parameters : arr, optional
            3D Lorentzian parameter array. By default None.
        path : list, optional
            List of paths to be drawn out. By default None.
        method : str, optional
            Method to be used for fitting any new Lorentzians that are
            selected. See the docstrings for `scipy.optimize.least_squares` for
            more information. By default 'lm'.
        """
        self.data_files = data_files
        if parameters is None:
            parameters = util.get_all_params(data_files)
        self.parameters = parameters
        self.method = method
        self.path = path
        self.setup_interface()

    def setup_interface(self):
        """
        Set up the tkinter GUI for interactive with the color plot session.
        """
        self.root = tk.Tk(
            baseName='interactiveSession',
            className='mistakeSelector')
        self.root.wm_title("Mistake Selector")
        self.controls = tk.Frame(self.root)
        self.controls.pack(side="top", fill="both", expand=False)
        self.make_button("Done",
            self.close_window,
            description="Finish tweaking and close window")
        self.entry = ttk.Combobox(
            master=self.root,
            takefocus=False,
            values=['Curve to Plot'] + [str(i)
                for i in range(len(self.parameters[0]))])
        self.entry.bind("<<ComboboxSelected>>", self.callback)
        self.entry.pack(in_=self.controls, side="left")
        self.entry.insert(0, 'Curve to Plot')
        self.make_button("Remove Selected",
            self.remove,
            description="Remove the currently displayed Lorentzian at the"
            " currently selected temperature")
        self.make_button("Refit Selected",
            self.refit,
            description="Refit the currently displayed Lorentzian at the"
            " currently selected temperature")
        self.make_button("Fill Gaps", self.fill_gaps)
        self.degree_tip = ttk.Label(
            master=self.root, text='Fill Degree  ', padding=3)
        self.degree = ttk.Entry(
            master=self.root,
            takefocus=False,
            width=2)
        self.degree.pack(in_=self.controls, side="left")
        self.degree_tip.pack(in_=self.controls, side="left")
        self.degree.insert(0, '1')
        self.overlap_tip = ttk.Label(
            master=self.root, text='Fill Overlap  ', padding=3)
        self.overlap = ttk.Entry(master=self.root, takefocus=False, width=2)
        self.overlap.pack(in_=self.controls, side="left")
        self.overlap_tip.pack(in_=self.controls, side="left")
        self.overlap.insert(0, '1')
        reg = self.root.register(_callback)
        self.degree.config(validate='key', validatecommand=(reg, '%P'))
        self.overlap.config(validate='key', validatecommand=(reg, '%P'))
        self.safe_bool = tk.BooleanVar()
        self.safe_check = ttk.Checkbutton(
            master=self.root,
            text='Safe Fill  ',
            onvalue=True,
            offvalue=False,
            variable=self.safe_bool,
            padding=3)
        self.safe_check.pack(in_=self.controls, side="left")
        self.make_button("Zoom In", self.zoom_in)
        self.make_button("Zoom Out", self.zoom_out)
        self.make_button("Reset Zoom", self.reset_zoom)
        self.make_button("Save", self.save)
        self.fig = Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        gs = GridSpec(3, 2)
        self.ax0 = self.fig.add_subplot(gs[0, 0])
        self.ax1 = self.fig.add_subplot(gs[0, 1])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[1, 1])
        self.ax4 = self.fig.add_subplot(gs[2, :])
        self.clear_axs()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.controls)
        self.toolbar.update()
        self.widget = self.canvas.get_tk_widget()
        self.widget.pack(side="top", fill="both", expand=True)
        self.canvas._tkcanvas.pack()
        self.canvas.mpl_connect('button_release_event', self.on_click)
        self.cs0 = Cursor(self.ax0, useblit=True, color='r',
                          linewidth=1, linestyle=":")
        self.cs1 = Cursor(self.ax1, useblit=True, color='r',
                          linewidth=1, linestyle=":")
        self.cs2 = Cursor(self.ax2, useblit=True, color='r',
                          linewidth=1, linestyle=":")
        self.cs3 = Cursor(self.ax3, useblit=True, color='r',
                          linewidth=1, linestyle=":")
        self.cs4 = RectangleSelector(
            self.ax4,
            self.on_select,
            drawtype='box',
            useblit=True,
            button=[1, 3],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
            rectprops=dict(
                facecolor='grey',
                edgecolor='green',
                alpha=0.2,
                fill=True))
        self.press = self.canvas.mpl_connect('key_press_event', self.on_press)
        self.p_ind = 0
        self.T_ind = 0
        self.extension = 2
        self.curve_plot(self.p_ind)
        self.mark_curves(self.T_ind)
        tk.mainloop()

    def save(self):
        """
        Save the current parameters as a pickled binary on your computer.
        """
        self.path = util.save(self.parameters)

    def autosave(self):
        """
        Autosave the current parameters as a pickled binary on your computer.
        """
        if self.path is not None:
            path = self.path[:-4] + '_autosave' + '.pkl'
            util.save(self.parameters, path)

    def new_curve(self):
        """
        Plot a new curve based on the one in the combobox in the upper left
        corner of the tkinter window.
        """
        s = self.entry.get()
        if s == 'Curve to Plot':
            pass
        else:
            self.clear_axs()
            self.p_ind = int(s)
            self.curve_plot(self.p_ind)

    def remove(self):
        """
        Set the value of the 1D Lorentzian currently displayed to all nans.
        This considers it "removed" from the overall Lorentzian parameter
        arrays.
        """
        self.correct_p([np.nan, np.nan, np.nan, np.nan])

    def zoom_out(self):
        """
        Zoom out as triggered by keyboard shortcut.
        """
        self.extension *= 1.5

    def zoom_in(self):
        """
        Zoom in as triggered by keyboard shortcut.
        """
        self.extension /= 1.5

    def reset_zoom(self):
        """
        Switch to the default zoom level as triggered by keyboard shortcut.
        """
        self.extension = 2

    def correct_p(self, p0):
        """
        Set the value of the 1D Lorentzian currently displayed to the given
        array of parameters.

        Parameters
        ----------
        p0 : arr
            1D Lorentzian parameter array.
        """
        p = self.parameters.astype("float")
        p[self.T_ind][self.p_ind] = p0
        self.parameters = np.array(p)
        self.mark_curves(self.T_ind)
        self.peak_plot(self.T_ind, self.p_ind)

    def correct_p_2d(self, p0):
        """
        Set the value of the 2D Lorentzian currently selected to the given
        array of parameters.

        Parameters
        ----------
        p0 : arr
            2D Lorentzian parameter array.
        """
        p = self.parameters.astype("float")
        for i in range(len(p)):
            p[i][self.p_ind] = p0[i]
        self.parameters = np.array(p)
        self.mark_curves(self.T_ind)
        self.peak_plot(self.T_ind, self.p_ind)

    def fill_gaps(self):
        """
        Fill all the gaps in the 2D Lorentzian currently selected.

        See Also
        --------
        util.find_missing_2d
        """
        self.autosave()
        old_params = self.parameters[:, self.p_ind]
        degree = self.degree.get()
        if len(degree) > 0:
            degree = int(degree)
        else:
            degree = 0
        overlap = self.overlap.get()
        if len(overlap) > 0:
            overlap = int(overlap)
        else:
            overlap = 1
        new_params = util.find_missing_2d(
            old_params,
            degree=degree,
            overlap=overlap,
            sanity_check=self.safe_bool)
        self.correct_p_2d(new_params)

    def refit(self, method=None):
        """
        Redo the fit on the 1D Lorentzian currently displayed.

        Parameters
        ----------
        method : str, optional
            Method used for fit, by default None.
        """
        self.autosave()
        if method is None:
            method = self.method
        f = self.data_files[self.T_ind].f
        v = self.data_files[self.T_ind].r
        min_ind = util.find_nearest_index(f, self.selection.f_min)
        max_ind = util.find_nearest_index(f, self.selection.f_max)
        region = np.array([[min_ind, max_ind]])
        p0 = fl.parameters_from_regions(
            f, v, region, max_n=1, force_fit=True, method=method)
        self.correct_p(p0)

    def curve_plot(self, p_ind):
        """
        Plot the curve at the provided index.

        Parameters
        ----------
        p_ind : int
            Selected index to plot the curve for.
        """
        self.autosave()
        self.ax0.plot(self.parameters[:, p_ind, 0], picker=True)
        self.ax1.plot(self.parameters[:, p_ind, 1], picker=True)
        self.ax2.plot(self.parameters[:, p_ind, 2], picker=True)
        self.ax3.plot(self.parameters[:, p_ind, 3], picker=True)
        self.ax0.relim()
        self.ax1.relim()
        self.ax2.relim()
        self.ax3.relim()
        self.ax0.autoscale_view()
        self.ax1.autoscale_view()
        self.ax2.autoscale_view()
        self.ax3.autoscale_view()
        self.canvas.draw_idle()

    def mark_curves(self, T_ind):
        """
        Draw the vertical red lines that mark which temperature is currently
        selected in the four Lorentzian component frames.

        Parameters
        ----------
        T_ind : int
            Index corresponding to the 1D Lorentzian currently displayed.
        """
        self.autosave()
        self.clear_curves()
        self.ax0.axvline(T_ind, color='r')
        self.ax1.axvline(T_ind, color='r')
        self.ax2.axvline(T_ind, color='r')
        self.ax3.axvline(T_ind, color='r')
        self.curve_plot(self.p_ind)
        self.canvas.draw_idle()

    def peak_plot(self, T_ind, p_ind):
        """
        Plot the specified peak.

        Parameters
        ----------
        T_ind : int
            Index corresponding to the 1D Lorentzian currently displayed.
        p_ind : int
            Index corresponding to the 2D Lorentzian currently selected.
        """
        self.autosave()
        self.ax4.cla()
        self.ax4.set_title('Preview')
        try:
            p = self.parameters[T_ind][p_ind]
            region = fl.regions_from_parameters(
                self.data_files[T_ind].f, [p], extension=self.extension)[0]
            region = [int(region[0]), int(region[1])]
            f_reg = self.data_files[T_ind].f[region[0]:region[1]]
            r_reg = self.data_files[T_ind].r[region[0]:region[1]]
            v_reg = gl.multi_lorentz_2d(f_reg, np.array([p]))
            v_reg += (np.mean(r_reg) - np.mean(v_reg))
            self.ax4.plot(f_reg, r_reg)
            self.ax4.plot(f_reg, v_reg, color='r')
            self.cs4.connect_default_events()
            self.canvas.draw_idle()
        except BaseException:
            try:
                better_T_ind = self.nearest_plot(T_ind, p_ind)
                p = self.parameters[better_T_ind][p_ind]
                region = fl.regions_from_parameters(
                    self.data_files[better_T_ind].f,
                    [p], extension=self.extension)[0]
                region = [int(region[0]), int(region[1])]
                f_reg = self.data_files[T_ind].f[region[0]:region[1]]
                r_reg = self.data_files[T_ind].r[region[0]:region[1]]
                v_reg = gl.multi_lorentz_2d(f_reg, np.array([p]))
                v_reg += (np.mean(r_reg) - np.mean(v_reg))
                self.ax4.plot(f_reg, r_reg)
                self.ax4.plot(f_reg, v_reg, color='r', alpha=0.2)
                self.cs4.connect_default_events()
                self.canvas.draw_idle()
            except BaseException:
                self.ax4.text(
                    0.5,
                    0.5,
                    'No parameters at this temperature.',
                    ha='center')
                self.cs4.disconnect_events()
                self.canvas.draw_idle()

    def nearest_plot(self, T_ind, p_ind, search='both'):
        """
        Find the nearest fit Lorentzian to the currently selected peak.

        Parameters
        ----------
        T_ind : int
            Index corresponding to the 1D Lorentzian currently displayed.
        p_ind : int
            Index corresponding to the 2D Lorentzian currently selected.
        search : str, optional
            [description], by default 'both'

        Returns
        -------
        int
            Index corresponding to the nearest 1D Lorentzian that has a fit to
            the 1D Lorentzian currently displayed. Note that if the Lorentzian
            currently displayed has a fit, then its index is the one returned.
            This index is always within the same `p_ind` as the current
            Lorentzian.
        """
        if T_ind < 0 or T_ind >= len(self.data_files):
            return None
        elif any(np.isnan(self.parameters[T_ind][p_ind])):
            if search == 'up':
                return self.nearest_plot(T_ind + 1, p_ind, search='up')
            elif search == 'down':
                return self.nearest_plot(T_ind - 1, p_ind, search='down')
            else:
                up_ind = self.nearest_plot(T_ind + 1, p_ind, search='up')
                down_ind = self.nearest_plot(T_ind - 1, p_ind, search='down')
                if up_ind is None:
                    up_ind = np.inf
                if down_ind is None:
                    down_ind = np.inf
                better_ind = min(up_ind, down_ind,
                                 key=lambda i: np.abs(i - T_ind))
                if better_ind >= len(self.data_files):
                    return None
                else:
                    return better_ind
        else:
            return T_ind

    def clear_curves(self):
        """
        Clear the curves from the four parameter frames.
        """
        self.autosave()
        self.ax0.cla()
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax0.set_title('Amplitude')
        self.ax1.set_title('Position')
        self.ax2.set_title('Full Width at Half Maximum')
        self.ax3.set_title('Phase')

    def clear_axs(self):
        """
        Clear all the axes in the tkinter window.
        """
        self.autosave()
        self.clear_curves()
        self.ax4.cla()
        self.ax4.set_title('Preview')

    def make_button(self, text, command, description=None):
        """
        Make a tkinter button and add it to the GUI.

        Parameters
        ----------
        text : string
            Text to show in the button.
        command : function
            Method to run when the button is pressed.
        description : string
            Text for the tooltip. By default None.
        """
        button = ttk.Button(master=self.root, text=text,
                            command=command, takefocus=False)
        button.pack(in_=self.controls, side="left")
        if description is not None:
            _Tooltip.create_tooltip(button, description)

    def close_window(self):
        """
        Safely disconnect canvas components and close the tkinter window.
        """
        self.autosave()
        self.canvas._tkcanvas.pack_forget()
        self.toolbar.pack_forget()
        self.canvas.mpl_disconnect(self.on_click)
        self.root.quit()
        self.root.destroy()

    def on_click(self, event):
        """
        Trigger for actions to be done upon picking a point or just clicking in
        general.

        Parameters
        ----------
        event : matplotlib event
            The click.
        """
        T_ind = int(np.round(event.xdata))
        self.widget.focus_set()
        if T_ind < len(self.data_files):
            self.T_ind = T_ind
            self.mark_curves(self.T_ind)
            try:
                self.peak_plot(self.T_ind, self.p_ind)
            except BaseException:
                self.ax4.cla()
                self.ax4.set_title('Preview')
                self.ax4.text(
                    0.5,
                    0.5,
                    'No parameters at this temperature.',
                    ha='center')

    def on_select(self, click, release):
        """
        Action to be done upon dragging out a selected window.

        Parameters
        ----------
        click : matplotlib event
            The event corresponding to the initial click.
        release : matplotlib event
            The event corresponding to when the click is released.
        """
        x1, y1 = click.xdata, click.ydata
        x2, y2 = release.xdata, release.ydata
        x_delta = np.abs(x1 - x2)
        y_delta = np.abs(y1 - y2)
        x_pos = min(x1, x2)
        y_pos = min(y1, y2)
        self.selection = _Selection(x_delta, y_delta, x_pos, y_pos)

    def callback(self, event):
        """
        Trigger new curve upon event.

        Parameters
        ----------
        event : matplotlib event
            The event to trigger off of.
        """
        self.new_curve()

    def on_press(self, event):
        """
        Parse keystrokes and apply the correct keyboard shortcuts.

        Parameters
        ----------
        event : Matplotlib event.
            The event recorded from the user's keystroke.
        """
        self.autosave()
        if event.key == 'enter':
            self.refit()
        elif event.key == '1':
            self.refit(method='trf')
        elif event.key == '2':
            self.refit(method='lm')
        elif event.key == '3':
            self.refit(method='dogbox')
        elif event.key == 'backspace':
            self.remove()
        elif event.key == 'right':
            self.T_ind += 1
            self.T_ind %= len(self.data_files)
            self.mark_curves(self.T_ind)
            self.peak_plot(self.T_ind, self.p_ind)
        elif event.key == 'left':
            self.T_ind -= 1
            self.T_ind %= len(self.data_files)
            self.mark_curves(self.T_ind)
            self.peak_plot(self.T_ind, self.p_ind)
        elif event.key == 'up':
            self.p_ind += 1
            self.p_ind %= len(self.parameters[0])
            self.entry.set(str(self.p_ind))
            self.mark_curves(self.T_ind)
            self.peak_plot(self.T_ind, self.p_ind)
        elif event.key == 'down':
            self.p_ind -= 1
            self.p_ind %= len(self.parameters[0])
            self.entry.set(str(self.p_ind))
            self.mark_curves(self.T_ind)
            self.peak_plot(self.T_ind, self.p_ind)
        elif event.key == 'escape':
            self.close_window()
        elif event.key == 's':
            self.save()
        elif event.key == 'o':
            self.zoom_out()
            self.mark_curves(self.T_ind)
            self.peak_plot(self.T_ind, self.p_ind)
        elif event.key == 'i':
            self.zoom_in()
            self.mark_curves(self.T_ind)
            self.peak_plot(self.T_ind, self.p_ind)
        elif event.key == 'h':
            self.reset_zoom()
            self.mark_curves(self.T_ind)
            self.peak_plot(self.T_ind, self.p_ind)
        elif event.key == 'f':
            self.fill_gaps()
            self.peak_plot(self.T_ind, self.p_ind)


def _check_contains(old_area, new_area):
    """
    See if a new region entirely contains an old one.

    Parameters
    ----------
    old_area : tuple
        A four element tuple of the form `(x_min, x_max, y_min, y_max)`.
    new_area : tuple
        A four element tuple of the form `(x_min, x_max, y_min, y_max)`.

    Returns
    -------
    bool
        Whether or not the old area is entirely contained in the new area.
    """
    checks = [
        old_area[0] >= new_area[0],
        old_area[1] <= new_area[1],
        old_area[2] >= new_area[2],
        old_area[3] <= new_area[3]
    ]
    return all(checks)


def color_selection(
        data_files,
        x_res=1000,
        y_res=100,
        cmap="viridis",
        parameters=None,
        method='lm',
        no_out=False):
    """
    Interactive Lorentzian tracker over a range of temperatures. Displays the
    peaks over all sweeps as a topographical color plot.

    Parameters
    ----------
    data_files : list
        List of data files to process and display.
    x_res : int, optional
        Initial resolution of the displayed x-axis. Default is `1000` but can
        be set higher or lower depending on what sort of graphics your computer
        can handle. The maximum value is the length of the frequency array in
        each data file. The x-axis is the frequency axis.
    y_res : int, optional
        Initial resolution of the displayed y-axis. Default is `100` but can be
        set higher or lower depending on what sort of graphics your computer
        can handle. The maximum value is the length of the list of data files
        inputted into `data_files`. The y-axis just corresponds to the sweeps
        in the order that they're given in the inputted list of data files. For
        a linear change in temperature this means the values displayed are
        proportional to the temperature. But, they are not the temperature
        itself.
    cmap : str, optional
        Initial colormap used for the displayed plot. Defaults to `viridis` but
        accepts any colormap that would be usable by `matplotlib`. This can be
        changed from a selection of other colormaps during the interactive
        plotting process.
    parameters : arr, optional
        Any pre-determined Lorentzian parameters to continue working from. The
        parameters returned by `color_selection` work for this purpose. This
        exists so you can save your work during the fitting process and load it
        again later.
    method : {'trf', 'dogbox', 'lm'}, optional
        Fitting method to use by the :func: `<scipy.optimize.least_squares>`
        backend for fitting to Lorentzians. See the documentation on that for
        more details. You generally do not have to change this.
    no_out : bool, optional
        Makes the default zoom slightly scaled down along the y-axis. This
        is to make it easier to use the rectangle zoom tool over a region
        without accidentially cutting out any data you need to fit over.

    Returns
    -------
    arr
        The 3D parameter array of the Lorentzians fitted from the inputted data
        files.
            - Axis 0 determines which sweep.
            - Axis 1 determines which Lorentzian.
            - Axis 2 is the parameters of the given Lorentzian.

    Shortcuts
    ---------
    a
        Activate tracer mode. This is helpful when you can see the peaks on the
        color plot but can't get a good fit on them. You can start with a trace
        and then adjust them to fit over the fittable regions in mistake
        selector mode.
    c
        Commit the current tracings to frequencies. The tracings are made in
        tracer mode. This produces Lorentzians for each trace. However, only
        the frequency is "really" physically meaningful. All the other
        parameters are guessed with rough heuristics based on the data
        surrounding the frequency.
    d
        Delete poorly fitted Lorentzians. When an entire curve is deleted in
        this way the peak is removed from the parameter array.
    e
        Toggle Enhance! mode. This lets you see an area in greater detail even
        if you're rendering with a low initial resolution.
    f
        Reload the canvas. Stands for "flush."
    h
        Return to "home" view. This resets the canvas to view all sweeps and
        frequencies.
    i
        Activate "inspire me" this runs a basic machine learning model from
        `automatic` that finds peaks on the selected sweep. This is useful when
        peaks are low amplitude and you're trying to find them without looking
        carefully over all frequencies by eye.
    k
        Lets you see the temperature of a given index. Just click on a spot on
        the color plot to see the temperature there. This shortcut is a toggle.
        So, press it again to turn temperature preview off.
    l
        Apply physically meaningful labels to the color plot's axes.
    m
        Stands for "move." Toggles the panning tool.
    o
        Enables trace smoothing. This is done using a Savitsky-Golay filter
        of order three over a window of five temperatures. These values are
        currently hard coded and cannot be changed.
    p
        Lets you select a particular sweep and opens a "paramter preview"
        window where you can interactively fit particular peaks as with
        `live_selection`. These are displayed on the color plot when done in
        order to let you know where peaks are for curve finding.
    q
        Quits the interactive color selector.
    r
        Renders the current selections.
    s
        Saves the currently fit parameters and enables auto-save so you don't
        have to worry about your work getting destroyed.
    t
        Enter "tweak" mode.
    u
        Captures a high resolution snapshot of the current image.
    x
        Toggles enhancements. This may help with certain rendering issues.
    y
        Prints some debugging info.
    z
        Toggles "zoom" tool.
    space
        Toggles display features.
    up
        Raises components in the live selector window or views the next peak in
        the mistake selector window.
    down
        Lowers components in the live selector window or views the previous
        peak in the mistake selector window.
    left
        Raise projection in the live selector window or move along the peak's
        path in the mistake selector window.
    right
        Lower projection in the live selector window or move along the peak's
        path in the mistake selector window.
    escape
        Restart the current selection or the drawing in Enhance! mode.
    control
        Edit the vertices in the current selection or draw area from the center
        in Enhance! mode.
    shift
        Draw a perfectly square area in Enhance! mode.
    enter
        Commit current selection for fitting.

    """
    y_res = min(y_res / len(data_files), 1)
    if parameters is not None and len(parameters) == 2:
        parameters = parameters[1]
    selector = _Color_Selector(data_files, x_res=x_res, y_res=y_res,
                               cmap=cmap, parameters=parameters, method=method,
                               no_out=no_out)
    return fl.parameters_from_selections(
        data_files, (selector.selections, selector.parameters), method=method)


def mistake_selection(data_files, parameters=None, path=None, method='lm'):
    """
    Interactive Lorentzian editor over a range of temperatures.

    Parameters
    ----------
    data_files : list
        List of data files to process and display.
    parameters : arr, optional
        Any pre-determined Lorentzian parameters to continue working from. The
        parameters returned by `color_selection` work for this purpose. This
        exists so you can save your work during the fitting process and load it
        again later.
    path : str, optional
        File path to use for autosaving.
    method : {'trf', 'dogbox', 'lm'}, optional
        Fitting method to use by the :func: `<scipy.optimize.least_squares>`
        backend for fitting to Lorentzians. See the documentation on that for
        more details. You generally do not have to change this.

    Returns
    -------
    arr
        The 3D parameter array of the Lorentzians fitted from the inputted data
        files.
            - Axis 0 determines which sweep.
            - Axis 1 determines which Lorentzian.
            - Axis 2 is the parameters of the given Lorentzian.

    Shortcuts
    ---------
    f
        Fill the gaps in the currently selected Lorentzian. This does not fill
        in the edge. For more, see `find_missing_2d` in `utilities`.
    h
        Return to "home" view. This resets the canvas to view all sweeps and
        frequencies.
    i
        Zoom in.
    o
        Zoom out.
    1
        Fit a Lorentzian using a Trust Region Reflective algorithm. See
        `<scipy.optimize.least_squares>`.
    2
        Fit a Lorentzian using a Levenberg-Marquardt algorithm. See
        `<scipy.optimize.least_squares>`.
    3
        Fit a Lorentzian using a Dogleg algorithm with rectangular trust
        regions. See `<scipy.optimize.least_squares>`.
    up
        View the next peak.
    down
        View the previous peak.
    left
        Pan to the left.
    right
        Pan to the right.
    enter
        Commit current selection for fitting.
    """
    m = _Mistake_Selector(data_files, parameters, path, method=method)
    return m.parameters


def _make_colors(data_files, max_res=1000, y_res=1, cmap="viridis"):
    """
    Generate a complete color plot for a given set of data_files.

    Parameters
    ----------
    data_files : list
        List of data_files.
    max_res : int, optional
        Total number of points rendered along the x_axis. Defaults to 1000.
        This is just the default resolution rendered. All frequency values
        are still included in the data and can be interacted with.
    y_res : int, optional
        Fractional resolution of the y_axis. Defaults to 1. This default means
        that all points along the y_axis (all temperatures/data_files) are
        rendered by default.
    cmap : str, optional
        Color map to render with, by default "viridis."

    Returns
    -------
    ax
        The axes of the rendered plot. Essential for the interactive matplotlib
        session.
    pts
        Points that correspond to each frequency value to be interacted with
        in the subplot. These are not visible to the user. However, they need
        to be rendered anyways so that they can be grabbed with the polygon
        selector generated by matplotlib. These are rendered with an opacity
        of zero.
    colors
        The colormesh for the rendered color plot.
    """
    x = np.empty((0,))
    y = np.empty((0,))
    z = np.empty((0, max_res))
    y_skip = int(np.round(1 / y_res))
    for i in range(0, len(data_files)):
        if i % y_skip == 0:
            z0 = data_files[i].r
            z0 = cd.scale_zoom(z0, 0, 1)
            z0 = cd.normalize_1d(z0, (min(z0), max(z0), max_res))
            coef = np.polyfit(np.arange(max_res), z0, 3)
            bg = np.poly1d(coef)
            z0 -= bg(np.arange(0, max_res))
            z = np.append(z, [z0], axis=0)
        x0 = np.arange(max_res)
        x = np.append(x, x0)
        y = np.append(y, np.ones(x0.shape) * i)
    color_x = np.linspace(0, max_res, max_res)
    color_y = np.linspace(0, len(data_files), len(z))
    fig = Figure()
    ax = fig.add_subplot(111)
    pts = ax.scatter(x, y, alpha=0)
    colors = ax.pcolormesh(color_x, color_y, z, cmap=cmap, shading='auto')
    return ax, pts, colors


def _points_to_ranges(points, data_files, res):
    """
    Find the minimum and maximum frequencies defined by the provided points as
    generated from a polygon selection in an interactive color plot. Then turn
    those points into ranges that can be sent into a fitting subroutine.

    Parameters
    ----------
    points : array-like
        Offsets for the invisible points to render in the selection. See
        `_make_colors` for more info.
    data_files : list
        List of data_files.
    res : int
        Resolution of the x_axis. Used for math about how the points are
        distributed to identify the corresponding frequencies.

    Returns
    -------
    dict
        A dictionary of ranges to fit over identified by the indices of the
        data_file containing the frequencies to be fit over within that range.

    See Also
    --------
    `_make_colors`
    """
    point_matching = []
    for i in range(0, len(data_files)):
        point_matching.append([])
    for i in range(0, len(points)):
        j = int(points[i][1])
        point_matching[j].append(points[i][0])
    range_dict = {}
    for i in range(0, len(point_matching)):
        # print(point_matching[i])
        try:
            min_pt = min(point_matching[i])
            max_pt = max(point_matching[i])
            min_val = min_pt / res
            max_val = max_pt / res
            f = data_files[i].f
            delta_f = max(f) - min(f)
            min_f = min(f) + (delta_f * min_val)
            max_f = min(f) + (delta_f * max_val)
            min_ind = util.find_nearest_index(f, min_f)
            max_ind = util.find_nearest_index(f, max_f)
            region = np.array([[min_ind, max_ind]])
            range_dict[i] = region
        except BaseException:
            pass
    return range_dict


def live_selection(data_file, params=None, vline=None):
    """
    Interactive peak selector at a single temperature.

    Parameters
    ----------
    data_file : Data_File
        The `Data_File` that with all the data to be looked at.
    params : arr, optional
        A 2D Lorentzian parameter array with any parameters to start with.
        This allows you to pause your work and continue later.
    vline : float, optional
        A specific frequency to mark with a vertical green line. This is done
        for integration with the interactive color plot primarily.

    Returns
    -------
    arr
        A 2D Lorentzian parameter array of all the peaks selected within the
        interactive session.
    """
    f = data_file.f
    x = data_file.x
    y = data_file.y
    r = data_file.r
    live = _Live_Instance(f, r)
    live.import_all_data(x, y)
    if params is not None:
        live.import_lorentzians(params)
    if vline is not None:
        live.set_vline(vline)
    live.activate()
    return live.get_all_params()


def point_selection(data_files, params=None, fs=[]):
    """
    An interactive session to help with finding redundant peaks between several
    RUS sweeps at the same given temperature.

    Parameters
    ----------
    data_files : list
        List of `data_files` at the same temperature.
    params : arr, optional
        A 3D Lorentzian parameter array correpsonding to the parameters for
        the given data_files. Also accepts a list of 2D Lorentzian parameter
        arrays.
    fs : list, optional
        Any already chosen frequencies to put markers out and include in the
        export.

    Returns
    -------
    list
        The final list of chosen frequencies.
    """
    fs = list(fs)
    if params is not None:
        util.set_all_params(data_files, params)
    selector = _Point_Selector(data_files, fs)
    return np.array(selector.chosen_frequencies)
