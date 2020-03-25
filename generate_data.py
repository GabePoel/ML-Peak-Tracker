import generate_lorentz as gl
import numpy as np
import os
# Generates 1000 sample data sets in new directory
export_dir = os.path.join(os.getcwd(), 'generated_data')
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
for i in range(0, 1000):
    num_str = str('000'[0:3 - len(str(i))]) + str(i)
    background_params, lorentz_params, f, displacement = gl.generate_data()
    data = np.transpose(np.vstack([f, displacement]))
    background_name = os.path.join(export_dir, num_str + '_background.csv')
    lorentz_name = os.path.join(export_dir, num_str + '_lorentz.csv')
    data_name = os.path.join(export_dir, num_str + '_data.csv')
    np.savetxt(background_name, background_params, delimiter=',')
    np.savetxt(lorentz_name, lorentz_params, delimiter=',')
    np.savetxt(data_name, data, delimiter=',')