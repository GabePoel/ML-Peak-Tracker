import numpy as np
import tkinter as tk
import PySimpleGUI as sg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import PolygonSelector
from matplotlib.widgets import Cursor
from matplotlib.patches import Polygon
from matplotlib.path import Path
from scipy.optimize import curve_fit
from . import fit_lorentz as fl
from . import generate_lorentz as gl
from . import classify_data as cd
from . import utilities as util

def lin(x, a, b):
    return x * b + a

def preview(f, v, params):
    live = Live_Instance(f, v)
    live.import_lorentzians(params)
    live.activate()

class Live_Lorentz():
    def __init__(self, p):
        self.A = p[0]
        self.f0 = p[1]
        self.FWHM = p[2]
        self.phase = p[3]

    def params(self):
        return np.array([self.A, self.f0, self.FWHM, self.phase])

class Selection():
    def __init__(self, x_delta, y_delta, x_pos, y_pos):
        self.x_delta = x_delta
        self.y_delta = y_delta
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.f_min = self.x_pos
        self.f_max = self.x_pos + self.x_delta

class Live_Instance():
    def __init__(self, f, v):
        self.f = f[np.logical_not(np.isnan(f))]
        self.v = v[np.logical_not(np.isnan(v))]
        self.live_lorentzians = {}
        self.all_data = False
        self.show_components = False
        self.component_height = 1.5
        self.projection_height = -1

    def import_all_data(self, x, y, data_to_analyze=None):
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
        for i in range(0, len(p_table)):
            p = p_table[i]
            f0 = p[1]
            self.live_lorentzians[f0] = Live_Lorentz(p)

    def activate(self, loop=True):
        self.first_load = True
        self.fig = Figure()
        self.fig.set_size_inches(16, 9)
        self.ax = self.fig.add_subplot(111)
        self.root = tk.Tk()
        self.root.wm_title("Live Peak Finder")
        self.controls = tk.Frame(self.root)
        self.controls.pack(side="top", fill="both", expand=True)
        if loop:
            self.make_button("Done", command=self.close_window)
            self.make_button("Refresh", command=self.update_window)
            self.make_button("Reset Axes", command=self.reset_axes)
            self.make_button("Add Lorentzians", command=self.add_lorentz)
            self.make_button("Remove Lorentzians", command=self.remove_lorentz)
            self.make_button("Show/Hide Components", command=self.components_bool)
            self.make_button("Raise Components", command=self.raise_components)
            self.make_button("Lower Components", command=self.lower_components)
            self.make_button("Raise Projection", command=self.raise_projection)
            self.make_button("Lower Projection", command=self.lower_projection)
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="bottom", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas._tkcanvas.pack()
        self.plot_lorentzians()
        if loop:
            tk.mainloop()

    def make_button(self, text, command):
        button = tk.Button(master=self.root, text=text, command=command)
        button.pack(in_=self.controls, side="left")

    def close_window(self):
        self.root.quit()
        self.root.destroy()

    def update_window(self):
        self.canvas._tkcanvas.pack_forget()
        self.toolbar.pack_forget()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="bottom", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas._tkcanvas.pack()

    def reset_axes(self):
        self.ax.set_xlim(self.default_x_lim)
        self.ax.set_ylim(self.default_y_lim)
        self.plot_lorentzians()

    def get_all_params(self):
        p_table = np.empty((0, 4))
        for l in self.live_lorentzians:
            p = self.live_lorentzians[l].params()
            p_table = np.append(p_table, np.array([p]), axis=0)
        return p_table

    def plot_lorentzians(self):
        if not self.first_load:
            x_lim = self.ax.get_xlim()
            y_lim = self.ax.get_ylim()
        self.ax.cla()
        p_table = self.get_all_params()
        if len(p_table) > 0:
            full_v = gl.multi_lorentz_2d(self.f, p_table)
        else:
            full_v = np.zeros((len(self.f),))
        offset = self.projection_height * np.abs(min(self.v) - max(self.v))
        for i in range(0, len(p_table)):
            ex_f, ex_v = fl.lorentz_bounds_to_data(p_table[i], self.f, self.v, expansion=2)
            self.ax.axvline(x=min(ex_f), color='pink')
            self.ax.axvline(x=max(ex_f), color='pink')
        self.ax.plot(self.f, self.v, color='b')
        self.ax.plot(self.f, full_v + offset, color='b')
        for i in range(0, len(p_table)):
            og_f, og_v = fl.lorentz_bounds_to_data(p_table[i], self.f, self.v, expansion=2)
            ex_f, ex_v = fl.lorentz_bounds_to_data(p_table[i], self.f, full_v, expansion=2)
            if self.all_data and self.show_components:
                small_f, small_x = fl.lorentz_bounds_to_data(p_table[i], self.f, self.x, expansion=2)
                x_opt, x_cov = curve_fit(lin, small_f, small_x)
                x_fit = lin(small_f, *x_opt)
                self.ax.plot(small_f, small_x - x_fit + np.mean(og_v) + self.component_height * (np.max(og_v) - np.min(og_v)), color='y')
                small_f, small_y = fl.lorentz_bounds_to_data(p_table[i], self.f, self.y, expansion=2)
                y_opt, y_cov = curve_fit(lin, small_f, small_y)
                y_fit = lin(small_f, *y_opt)
                self.ax.plot(small_f, small_y - y_fit + np.mean(og_v) + self.component_height * (np.max(og_v) - np.min(og_v)), color='g')
            self.ax.plot(og_f, og_v, color='r')
            self.ax.plot(ex_f, ex_v + offset, color='r')
        if not self.first_load:
            self.ax.set_xlim(x_lim)
            self.ax.set_ylim(y_lim)
        else:
            self.default_x_lim = self.ax.get_xlim()
            self.default_y_lim = self.ax.get_ylim()
        self.first_load = False
        self.update_window()

    def add_lorentz(self):
        self.end_interactive()
        self.start_add_interactive()
        self.act_press = self.fig.canvas.mpl_connect('key_press_event', self.on_press_add)

    def remove_lorentz(self):
        self.end_interactive()
        self.start_rem_interactive()
        self.act_press = self.fig.canvas.mpl_connect('key_press_event', self.on_press_rem)

    def start_add_interactive(self):
        self.selection = None
        self.cursor = Cursor(self.ax, useblit=True, color='0.5', linewidth=1, linestyle=":")
        self.act_select = RectangleSelector(self.ax, self.on_select, 
            drawtype='box', 
            useblit=False, 
            button=[1, 3], 
            minspanx=5, 
            minspany=5, 
            spancoords='pixels', 
            interactive=True,
            rectprops = dict(facecolor='grey', edgecolor = 'green', alpha=0.2, fill=True))

    def start_rem_interactive(self):
        self.selection = None
        self.cursor = Cursor(self.ax, useblit=True, color='0.5', linewidth=1, linestyle=":")
        self.act_select = RectangleSelector(self.ax, self.on_select, 
            drawtype='box', 
            useblit=False, 
            button=[1, 3], 
            minspanx=5, 
            minspany=5, 
            spancoords='pixels', 
            interactive=True,
            rectprops = dict(facecolor='grey', edgecolor = 'red', alpha=0.2, fill=True))

    def end_interactive(self):
        try:
            self.canvas.mpl_disconnect(self.act_press)
        except:
            pass
        try:
            self.act_select.set_active(False)
        except:
            pass
        self.plot_lorentzians()

    def on_press_add(self, event):
        if event.key == 'enter':
            min_ind = util.find_nearest_index(self.f, self.selection.f_min)
            max_ind = util.find_nearest_index(self.f, self.selection.f_max)
            region = np.array([[min_ind, max_ind]])
            region_f, region_v = util.extract_region(0, region, self.f, self.v)
            p_list = [fl.free_n_least_squares(region_f, region_v, max_n=1, force_fit=True).x]
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

    def on_press_rem(self, event):
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

    def on_select(self, click, release):
        x1, y1 = click.xdata, click.ydata
        x2, y2 = release.xdata, release.ydata
        x_delta = np.abs(x1 - x2)
        y_delta = np.abs(y1 - y2)
        x_pos = min(x1, x2)
        y_pos = min(y1, y2)
        self.selection = Selection(x_delta, y_delta, x_pos, y_pos)

    def components_bool(self):
        self.show_components = not self.show_components
        self.plot_lorentzians()

    def raise_components(self):
        self.component_height += 0.5
        self.plot_lorentzians()

    def lower_components(self):
        self.component_height -= 0.5
        self.plot_lorentzians()

    def raise_projection(self):
        self.projection_height += 0.5
        self.plot_lorentzians()

    def lower_projection(self):
        self.projection_height -= 0.5
        self.plot_lorentzians()

class Color_Selector:
    """
    Literally just the SelectFromCollection example.
    """
    def __init__(self, data_files, alpha_other=0.0, x_res=1000, y_res=1, cmap='cool'):
        self.x_res = x_res
        self.cmap = cmap
        self.data_files = data_files
        ax, collection = make_colors(data_files, max_res=x_res, y_res=y_res, cmap=cmap)
        self.fig = ax.figure
        self.fig.set_size_inches(16, 9)
        self.ax = ax
        self.root = tk.Tk()
        self.root.wm_title("Color Selector")
        self.controls = tk.Frame(self.root)
        self.controls.pack(side="top", fill="both", expand=True)
        self.make_button("Done", command=self.close_window)
        self.make_button("Another!", command=self.another_selection)
        self.make_button("Toggle Selections", command=self.toggle_show)
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="bottom", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas._tkcanvas.pack()
        self.collection = collection
        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        self.poly = PolygonSelector(ax, self.on_select)
        self.press = self.canvas.mpl_connect('key_press_event', self.on_press)
        self.ind = []
        self.selections = []
        self.patches = []
        self.selectors = []
        self.show_patches = True
        self.slight_zoom_out()
        tk.mainloop()

    def reload(self):
        self.canvas._tkcanvas.pack_forget()
        ax, collection = make_colors(self.data_files, max_res=self.x_res, cmap=self.cmap)
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
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.canvas.draw_idle()

    def on_press(self, event):
        if event.key == 'enter':
            self.another_selection()
        elif event.key == ' ':
            self.toggle_show()

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()

    def make_button(self, text, command):
        button = tk.Button(master=self.root, text=text, command=command)
        button.pack(in_=self.controls, side="left")

    def close_window(self):
        try:
            self.another_selection()
        except:
            pass
        self.canvas._tkcanvas.pack_forget()
        self.toolbar.pack_forget()
        self.disconnect()
        self.canvas.mpl_disconnect(self.press)
        self.root.quit()
        self.root.destroy()

    def finish_selection(self):
        selection = points_to_ranges(np.array(self.xys[self.ind]), self.data_files, self.x_res)
        self.selections.append(selection)
        patch = Polygon(self.poly.verts, facecolor='b', alpha=0.3)
        self.patches.append(patch)
        self.disconnect()
        self.selectors.append(self.poly)
        self.ax.add_patch(patch)

    def another_selection(self):
        self.finish_selection()
        self.poly = PolygonSelector(self.ax, self.on_select)
        self.ind = []

    def toggle_show(self):
        self.show_patches = not self.show_patches
        if self.show_patches:
            for patch in self.patches:
                patch.set_alpha(0)
            for selector in self.selectors:
                selector.set_visible(False)
        else:
            for patch in self.patches:
                patch.set_alpha(0.3)
            for selector in self.selectors:
                selector.set_visible(True)
        self.canvas.draw_idle()

    def slight_zoom_out(self):
        # x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()
        # x_delta = x_lim[1] - x_lim[0]
        y_delta = y_lim[1] - y_lim[0]
        # new_x_lim = (x_lim[0] - 0.1 * x_delta, x_lim[1] + 0.1 * x_delta)
        new_y_lim = (y_lim[0] - 0.1 * y_delta, y_lim[1] + 0.1 * y_delta)
        # self.ax.set_xlim(new_x_lim[0], new_x_lim[1])
        self.ax.set_ylim(new_y_lim[0], new_y_lim[1])

def color_selection(data_files, x_res=1000, y_res=100, cmap='cool'):
    y_res = min(y_res / len(data_files), 1)
    selector = Color_Selector(data_files, x_res=x_res, y_res=y_res, cmap=cmap)
    return selector.selections

def make_colors(data_files, max_res=1000, y_res=1, cmap='cool'):
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
    ax.pcolormesh(color_x, color_y, z, cmap=cmap)
    return ax, pts

def points_to_ranges(points, data_files, res):
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
        except:
            pass
    return range_dict
