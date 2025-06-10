""" Convert the collected HDF5 files to Geotiff maps
0.5 hour but varies from a few minutes to ~1 hour per chunk
"""
import os
import fiona
import rasterio as rio
import numpy as np
import pandas as pd
from tqdm import tqdm
import tables
from process_gNATSGO_utils import *
from constants import path_intrim, path_data


gdb_path = os.path.join(path_data, 'gNATSGO_CONUS', 'gNATSGO_CONUS.gdb')
dst_dir = os.path.join(path_intrim, 'gNATSGO')


# The top 10 layers in ELM are hydrologically active
# here are the layers' interface depths, unit: m
layer_nodes = np.array([0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 
                        1.0380, 1.7276, 2.8646])


# Get the mapping from map unit key (mukey) to chunk number (1-73) of the divided geotif files
mukey_lkey = get_mukey_chunk(gdb_path, dst_dir)


chunk = REPLACE # 1-72

#-----------------------------------------------------------
# Map the chorizon data to geotiff files
# This also takes ~2 hours
#-----------------------------------------------------------
for var in ['sandtotal_r', 'silttotal_r', 'claytotal_r', 'om_r', 'ksat_r']:
    # Read the mukey level data table
    filename = os.path.join(dst_dir, f'mukey_data_{var}.h5')
    with tables.open_file(filename, mode='r') as f:
        df = pd.DataFrame(f.get_node(f'/{var}').read()).set_index('mukey')

    #for chunk in range(1, 73): 
    print(var, chunk)

    # Read the geotiff file
    with rio.open(os.path.join(dst_dir, f'mukey_grid_{chunk}.tif')) as src:
        profile = dict(src.profile)
        mukey_map = src.read(1)

    # Map the data table to the geotiff file
    profile['count'] = len(layer_nodes)
    profile['dtype'] = np.float32
    profile['compress'] = 'lzw'
    with rio.open(os.path.join(dst_dir, f'mukey_{var}_{chunk}.tif'),
                    'w', **profile) as dst:
        data = []
        for i in range(len(layer_nodes)):
            data.append( np.full(mukey_map.shape, np.nan) )

        for mukey in tqdm( mukey_lkey.index[mukey_lkey[f'grid_{chunk}']] ):
            if not mukey in df.index:
                continue
            filter = mukey_map == mukey
            for i in range(len(layer_nodes)):
                val = df.loc[mukey, f'elm_layer_{i}']
                data[i] = np.where(filter, val, data[i])

        for i in range(len(layer_nodes)):
            dst.write(data[i], i + 1)
            dst.update_tags(i + 1, BAND_NAME = f'elm_layer_{i}')


#-----------------------------------------------------------
# Put the bedrock depth into geotif files directly
#-----------------------------------------------------------
print('brockdepmin', chunk)

bedrock_data = []
src_gdb = fiona.open(gdb_path, layer = 'muaggatt')
for i, feat in enumerate(src_gdb):
    mukey = np.int32(feat['properties']['mukey'])
    brock = feat['properties']['brockdepmin']
    if not brock is None:
        # cm => m
        bedrock_data.append([mukey, brock / 100.])
bedrock_data = pd.DataFrame(bedrock_data)
bedrock_data.columns = ['mukey', 'brockdepmin']
bedrock_data = bedrock_data.set_index('mukey')

mukey_subset = mukey_lkey.loc[mukey_lkey.index.intersection(bedrock_data.index), :]

# Map the data table to the geotiff file
#for chunk in range(1, 73):
with rio.open(os.path.join(dst_dir, f'mukey_grid_{chunk}.tif')) as src:
    profile = dict(src.profile)
    mukey_map = src.read(1)
profile['count'] = len(layer_nodes)
profile['dtype'] = np.float32
profile['compress'] = 'lzw'
with rio.open(os.path.join(dst_dir, f'mukey_brockdepmin_{chunk}.tif'),
                'w', **profile) as dst:
    data = np.full(mukey_map.shape, np.nan)
    for mukey in tqdm( mukey_lkey.index[mukey_lkey[f'grid_{chunk}']] ):
        if not mukey in bedrock_data.index:
            continue
        data = np.where(mukey_map == mukey, bedrock_data.loc[mukey, 'brockdepmin'], data)
    dst.write(data, 1)

src_gdb.close()