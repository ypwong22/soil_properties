""" Make a loop through all the files and check the size > 0"""
import pandas as pd
import os

n_chunks = 72

var_list = ['sandtotal_r', 'silttotal_r', 'claytotal_r', 'om_r', 'ksat_r', 'brockdepmin']
generated = pd.DataFrame(False, index = range(1, n_chunks+1), columns = var_list)
path_folder = os.path.join(os.environ['PROJDIR'], 'DATA', 'Soil_Properties', 
                           'gNATSGO_CONUS', 'mukey_divide')

for var in var_list:
    for n in range(1, n_chunks+1):
        filename = os.path.join(path_folder, f'mukey_{var}_{n}.tif')
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            generated.loc[n, var] = True

generated.to_csv(os.path.join(path_folder, 'mukey_variables_checksum.csv'))
