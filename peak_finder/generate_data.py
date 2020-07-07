import numpy as np
import os
try:
    from . import generate_lorentz as gl
except:
    import generate_lorentz as gl

# Generates complete data sets with all the Lorentzians, background, parameters, etc. listed.
# Generally overkill for model training. Use efficient_data_generation.

def mag_round(x, base=10):
    """
    Rounds a number to the nearest order of magnitude.
    
    Parameters
    ----------
    x : float
        The number to be rounded.
    base : int
        The base that the number should be rounded with respect to.

    Returns
    -------
    float
        The number rounded to the nearest order of magnitude.
    """
    val = 1
    while x > val:
        val *= base
    return val

def get_append(index, count):
    """
    Gives a string to use to the end of large data sets with a bunch of files.
    """
    str_length = len(str(mag_round(count))) - 1
    base_str = str_length * '0'
    num_str = str(base_str[0:str_length - len(str(index))]) + str(index)
    return num_str

def generate_data_set(location=os.getcwd(), name='generated_data', count=1000, include_noise=True):
    """
    Generates (by default) 1000 sample data sets in new directory.
    """
    export_dir = os.path.join(location, name)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    for i in range(0, count):
        num_str = get_append(i, count)
        background_params, lorentz_params, f, displacement = gl.generate_data(include_noise=include_noise)
        data = np.transpose(np.vstack([f, displacement]))
        background_name = os.path.join(export_dir, num_str + '_background.csv')
        lorentz_name = os.path.join(export_dir, num_str + '_lorentz.csv')
        data_name = os.path.join(export_dir, num_str + '_data.csv')
        np.savetxt(background_name, background_params, delimiter=',')
        np.savetxt(lorentz_name, lorentz_params, delimiter=',')
        np.savetxt(data_name, data, delimiter=',')

def load_data_set(directory=os.path.join(os.getcwd(), 'generated_data')):
    """
    Loads all data from the specified directory.
    Returns as tuple of lists of background arrays, lorentz arrays, and data arrays.
    """
    file_names = os.listdir(directory)
    file_names.sort()
    background_paths = []
    lorentz_paths = []
    data_paths = []
    for i in range(0, len(file_names)):
        name_partition = file_names[i].split('_', 1)
        if name_partition[1] == 'background.csv':
            background_paths.append(file_names[i])
        elif name_partition[1] == 'lorentz.csv':
            lorentz_paths.append(file_names[i])
        elif name_partition[1] == 'data.csv':
            data_paths.append(file_names[i])
    count = len(data_paths)
    background_arrays = []
    lorentz_arrays = []
    data_arrays = []
    for i in range(0, count):
        background_file = os.path.join(directory, background_paths[i])
        lorentz_file = os.path.join(directory, lorentz_paths[i])
        data_file = os.path.join(directory, data_paths[i])
        background_arrays.append(np.genfromtxt(background_file, delimiter=','))
        lorentz_arrays.append(np.atleast_2d(np.genfromtxt(lorentz_file, delimiter=',')))
        data_arrays.append(np.genfromtxt(data_file, delimiter=','))
    return (background_arrays, lorentz_arrays, data_arrays)