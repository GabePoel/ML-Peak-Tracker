import classify_data as cd
import generate_data as gd
import generate_lorentz as gl
import numpy as np
import time
import sys
import os
import shutil
import zipfile
import utilities
from utilities import progressbar

# Makes smaller data sets than generate_data that provide the minimum information needed for model fitting.
# Generally use this unless you need everything else that generate_data provides.

def make_massive_data_set(number, scale=(0,1,1024), noise=True):
    background_arrays_list = []
    data_arrays_list = []
    lorentz_list = []
    for i in progressbar(range(number), "Generating Data: ", 40):
        background_params, lorentz_params, f, v = gl.generate_data(noise)
        background_arrays_list.append(background_params)
        lorentz_list.append(lorentz_params)
        data_arrays_list.append(cd.merge_data(f, v))
    return (background_arrays_list, lorentz_list, data_arrays_list)

def associate_lorentz(lorentz_array, index):
    associated_lorentz_array = np.empty((0,5))
    for i in range(0, lorentz_array.shape[0]):
        associated_lorentz = np.insert(lorentz_array[i], 0, index)
        associated_lorentz_array = np.append(associated_lorentz_array, np.array([associated_lorentz]), axis=0)
    return associated_lorentz_array

def make_simple_data_set(number, scale=(0,1,1024), noise=True):
    all_data = np.empty((0,scale[2]))
    all_lorentz = np.empty((0,5)) # Associated Data, A, f0, FWHM, Phase
    for i in progressbar(range(number), "Generating Data: ", 40):
        background_params, lorentz_params, f, v = gl.generate_data(noise)
        old_f_scale = cd.scale_1d(f)
        old_v_scale = cd.scale_1d(v)
        v_norm = cd.normalize_1d(v, scale)
        l_norm = cd.normalize_lorentz_2d(lorentz_params, old_f_scale, old_v_scale, scale, scale)
        l_associated = associate_lorentz(l_norm, i)
        all_data = np.append(all_data, np.array([v_norm]), axis=0)
        all_lorentz = np.append(all_lorentz, l_associated, axis=0)
    return (all_lorentz, all_data)

def make_blank_data_set(number, scale=(0,1,1024), noise=True):
    all_data = np.empty((0,scale[2]))
    for i in progressbar(range(number), "Generating Data: ", 40):
        background_params, lorentz_params, f, v = gl.generate_data(noise)
        v_norm = cd.normalize_1d(v, scale)
        all_data = np.append(all_data, np.array([v_norm]), axis=0)
    return (None, all_data)

def export_simple_data_set(simple_data_set, location=os.getcwd(), name='simple_set'):
    export_dir = os.path.join(location, name)
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)
    lorentz_path = os.path.join(export_dir, 'lorentz.csv')
    data_path = os.path.join(export_dir, 'data.csv')
    np.savetxt(lorentz_path, simple_data_set[0], delimiter=',')
    np.savetxt(data_path, simple_data_set[1], delimiter=',')
    shutil.make_archive(export_dir, 'zip', export_dir)
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

def import_simple_data_set(path):
    open_dir = os.path.join(os.getcwd(), 'temp_zip_dir')
    if not os.path.exists(open_dir):
        os.makedirs(open_dir)
    lorentz_path = os.path.join(open_dir, 'lorentz.csv')
    data_path = os.path.join(open_dir, 'data.csv')
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(open_dir)
    all_lorentz = np.genfromtxt(lorentz_path, delimiter=',')
    all_data = np.genfromtxt(data_path, delimiter=',')
    shutil.rmtree(open_dir)
    return (all_lorentz, all_data)

def convert_simple_data_set(simp):
    background_arrays_list = None
    data_arrays_list = []
    lorentz_arrays_list = []
    for i in progressbar(range(0, simp[1].shape[0]), "Converting Displacements: ", 40):
        data_arrays_list.append(cd.merge_data(np.linspace(0,1,1024), simp[1][i]))
        lorentz_arrays_list.append(np.empty((0,4)))
    for i in progressbar(range(0, simp[0].shape[0]), "Converting Lorentzians: ", 40):
        l_full = simp[0][i]
        association = int(l_full[0])
        l = l_full[1:5]
        l_arr = lorentz_arrays_list[association]
        l_arr = np.append(l_arr, np.array([l]), axis=0)
        lorentz_arrays_list[association] = l_arr
    return (background_arrays_list, lorentz_arrays_list, data_arrays_list)

def make_single_data_set(number, scale=(0,1,1024), noise=True):
    all_data = np.empty((0, scale[2]))
    all_lorentz = np.empty((0, 1))
    F = np.linspace(scale[0], scale[1], scale[2])
    for i in progressbar(range(number), "Generating Data: ", 40):
        has_lorentz = 1
        v, params = gl.generate_lorentz(F)
        if noise:
            v += gl.generate_noise(F, amount=1)
        else:
            v += gl.generate_noise(F, amount=0)
        f = np.linspace(scale[0], scale[1], scale[2])
        old_f_scale = cd.scale_1d(f)
        old_v_scale = cd.scale_1d(v)
        np.putmask(f, F < params[0][1] - (2 * params[0][2]), f * 0 - 1)
        np.putmask(f, F > params[0][1] + (2 * params[0][2]), f * 0 - 1)
        v = v[f > 0]
        f = f[f > 0]
        if np.random.random() > .5:
            v -= gl.multi_lorentz_2d(f, params)
            has_lorentz = 0
            if not noise:
                v = np.linspace(scale[0], scale[1], scale[2]) * (np.random.random() - 0.5) * 2
        v_norm = cd.normalize_1d(v, scale)
        if noise and has_lorentz == 1:
            v_norm += np.linspace(scale[0], scale[1], scale[2]) * (np.random.random() - 0.5) * 2
        v_norm += ((np.linspace(scale[0], scale[1], scale[2]) + (np.random.random() - 0.5) * 30) ** 2) * (np.random.random() - 0.5) * .5
        v_norm = cd.normalize_1d(v_norm, scale)
        all_data = np.append(all_data, np.array([v_norm]), axis=0)
        all_lorentz = np.append(all_lorentz, np.array([[has_lorentz]]), axis=0)
    return (all_lorentz, all_data)