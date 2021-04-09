import tkinter as tk
from tkinter import messagebox


def quick_analyze(f=None, v=None, show=True, learn=True):
    """
    Error message for analyzing when Google doesn't update Tensorflow.
    """
    tk.Tk().withdraw()
    messagebox.showerror("Can't do Machine Learning",
                         'Tensorflow not installed.')
