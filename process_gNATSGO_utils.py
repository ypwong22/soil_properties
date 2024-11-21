import os
import geopandas as gpd
import rasterio as rio
import numpy as np
from utils import vert_interp
import pandas as pd
from tqdm import tqdm
import tables


def get_mukey_chunk(gdb_path, dst_dir):
    ############################################################
    # Relationship between lkey (the SAPOLYGON)
    # and mukey (mapunit key)
    #
    # Each SAPOLYGON contains many map units, each map unit
    # contain multiple components, which may be vertial horizons
    ############################################################
    gdf = gpd.read_file(gdb_path, layer='mapunit')
    mukey_lkey = gdf[['mukey', 'lkey']]
    mukey_lkey = mukey_lkey.set_index('mukey')
    mukey_lkey.index = mukey_lkey.index.astype(np.int32)


    """
    Get the list of mukeys in each region, append to the mukey_lkey table.
    ~ takes a little over 5 min
    """
    for chunk in range(1, 73):
        mukey_lkey[f'grid_{chunk}'] = pd.Series(False, index = mukey_lkey.index, dtype = bool)
    for chunk in range(1, 73):

        # Get the list of mukeys in this piece
        with rio.open(os.path.join(dst_dir, f'mukey_grid_{chunk}.tif')) as src:
            mukey_map = src.read(1)

        alist = np.unique(mukey_map)
        if alist[-1] <= 0:
            continue

        i = 0
        while alist[i] <= 0:
            i = i + 1
        alist = alist[i:]

        mukey_lkey.loc[alist, f'grid_{chunk}'] = True

    # some mukey do not have a grid, perhaps because they do not exist
    mukey_lkey = mukey_lkey.loc[ \
        mukey_lkey[[f'grid_{chunk}' for chunk in range(1,73)]].sum(axis = 1) > 0, :]

    return mukey_lkey


def get_mukey_cokey(gdb_path):
    """ Relationship between cokey ("components" inside map unit)
        and mukey (mapunit key)"""
    mukey_cokey = gpd.read_file(gdb_path, layer='component')
    mukey_cokey = mukey_cokey[['comppct_r','mukey','cokey']].set_index('cokey')
    mukey_cokey['mukey'] = mukey_cokey['mukey'].astype(int)
    mukey_cokey['comppct_r'] = mukey_cokey['comppct_r'].astype(float)
    return mukey_cokey


def interpolate_cokey(co_depths, co_values, layer_nodes, layer_depth):
    """ Interpolate the entries that have the same cokey to 
        standard ELM depths """

    # reorder into ascending depth
    sort_ind = np.argsort(co_depths[:, 0])
    input_depths = co_depths[sort_ind, :]
    input_values = co_values[sort_ind]

    # (assume the input_interfaces are continuous)
    input_nodes = np.mean(input_depths, axis = 1)
    input_interfaces = np.insert(input_depths[:,1], 0, input_depths[0,0])

    input_values = input_values.reshape(1, -1)

    # interpolate to ELM depths
    entry_cokey = vert_interp(layer_nodes, input_nodes, input_values, False, 
                              layer_depth, input_interfaces)

    return entry_cokey


class mukeyWriter:
    """ Save the mukey-level entry to HDF5 file """

    def __init__(self, dst_dir: str, var: str, nlayers: int):
        self.dst_dir = dst_dir
        self.var = var
        self.nlayers = nlayers

        # create data tables if they do not exist
        # columns are ELM layers
        # rows are mukeys
        filename = os.path.join(dst_dir, f'mukey_data_{self.var}.h5')
        with tables.open_file(filename, 'w') as f:
            f.create_table(f.root, self.var, 
                dict([('mukey', tables.Int32Col())] + \
                     [(f'elm_layer_{i}', tables.Float32Col()) for i \
                       in range(self.nlayers)]))

    def insert(self, mukey, entry_mukey):
        filename = os.path.join(self.dst_dir, f'mukey_data_{self.var}.h5')
        # insert the data to the corresponding row in the hdf5 table
        # column: ELM layers, rows: variables
        with tables.open_file(filename, mode='a') as f:
            df = f.get_node(f'/{self.var}')
            row = df.row
            row['mukey'] = mukey
            for j in range(self.nlayers):
                row[f'elm_layer_{j}'] = entry_mukey[j]
            row.append()


def write_mukey_hdf5(var, variable_data, layer_nodes, layer_depth, dst_dir):
    """ Interpolate the cokey level data to ELM depths, weight-average to mukey level, and
        use the writer to save the mukey-level entries to HDF5 file """
    writer = mukeyWriter(dst_dir, var, len(layer_nodes))

    # iterate through map key
    for mukey in tqdm(variable_data.index.levels[0]):
        mukey_subset = variable_data.loc[mukey, :]

        # iterate through component key to interpolate to ELM depths
        comppct = []
        entry_cokey = []
        for cokey in np.unique(mukey_subset.index):
            cokey_subset = mukey_subset.loc[cokey, :]
            if len(cokey_subset.shape) == 1:
                # 1D data results in numeric value
                comppct.append(cokey_subset['copct'])
                # there is only one measurement, apply to all depths
                entry_cokey.append(
                    np.ones(len(layer_nodes)) * cokey_subset[var]
                )
            else:
                comppct.append(cokey_subset['copct'].iloc[0])
                entry_cokey.append( interpolate_cokey(
                    cokey_subset[['hzdept_r', 'hzdepb_r']].values / 100., 
                    cokey_subset[var].values, 
                    layer_nodes, layer_depth
                ) )
        entry_cokey = np.vstack(entry_cokey)
        comppct = np.array(comppct).reshape(1,-1)

        # sum up the components using weighted average
        entry_mukey = np.matmul(comppct, entry_cokey)[0, :] / np.sum(comppct)

        # append to HDF5 file
        writer.insert(mukey, entry_mukey)