import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import Cursor
from . import fit_lorentz as fl
from . import generate_lorentz as gl
from . import utilities as util

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
        self.f = f
        self.v = v
        self.live_lorentzians = {}
    
    def import_lorentzians(self, p_table):
        for i in range(0, len(p_table)):
            p = p_table[i]
            f0 = p[1]
            self.live_lorentzians[f0] = Live_Lorentz(p)

    def activate(self):
        self.first_load = True
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.root = tk.Tk()
        self.root.wm_title("Live Peak Finder")
        self.controls = tk.Frame(self.root)
        self.controls.pack(side="top", fill="both", expand=True)
        self.make_button("Done", command=self.close_window)
        self.make_button("Refresh", command=self.update_window)
        self.make_button("Reset Axes", command=self.reset_axes)
        self.make_button("Add Lorentzians", command=self.add_lorentz)
        self.make_button("Remove Lorentzians", command=self.remove_lorentz)
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="bottom", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas._tkcanvas.pack()
        self.plot_lorentzians()
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
        full_v = gl.multi_lorentz_2d(self.f, p_table)
        offset = 2 * min(self.v) - max(self.v)
        for i in range(0, len(p_table)):
            ex_f, ex_v = fl.lorentz_bounds_to_data(p_table[i], self.f, self.v, expansion=2)
            self.ax.axvline(x=min(ex_f), color='pink')
            self.ax.axvline(x=max(ex_f), color='pink')
        self.ax.plot(self.f, self.v, color='b')
        self.ax.plot(self.f, full_v + offset, color='b')
        for i in range(0, len(p_table)):
            og_f, og_v = fl.lorentz_bounds_to_data(p_table[i], self.f, self.v, expansion=2)
            ex_f, ex_v = fl.lorentz_bounds_to_data(p_table[i], self.f, full_v, expansion=2)
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

