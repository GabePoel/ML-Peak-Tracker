import numpy as np
import scipy as sp
import generate_lorentz as gl

# Will eventually have all the fitting stuff for final predictions.
# Currently just guestimates some heuristic paramaters to start the fit from.

def estimate_parameters(f, v, n=1):
    max_f = max(f)
    min_f = min(f)
    max_v = max(v)
    min_v = min(v)
    guess_FWHM = np.ones((1, n)) * (max_f - min_f) / n
    guess_f0 = (np.arange(n) + 1) * (max_f - min_f) / (n + 1) + min_f
    guess_A = np.ones((1, n)) * (max_v - min_v) / n
    guess_phase = np.ones((1, n)) * np.pi
    guess_params = np.transpose(np.concatenate([guess_A, guess_f0, guess_FWHM, guess_phase], axis=0))
    return guess_params