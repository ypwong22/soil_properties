""" Convert the gNATSGO tables to mapunit level and write to HDF5 tables 
~0.5 hrs per variable x 6 variables = <4 hours
"""
import os
import geopandas as gpd
import fiona
import numpy as np
import pandas as pd
from tqdm import tqdm
from process_gNATSGO_utils import *


gdb_path = os.path.join(
    os.environ['PROJDIR'], 'DATA', 'Soil_Properties', 'gNATSGO_CONUS', 'gNATSGO_CONUS.gdb'
)

dst_dir = os.path.join(os.environ['PROJDIR'], 'DATA', 'Soil_Properties', 
                       'gNATSGO_CONUS', 'mukey_divide')

# The top 10 layers in ELM are hydrologically active
# here are the layers' interface depths, unit: m
layer_depth = np.array([0, 0.0175, 0.0451, 0.0906, 0.1655, 0.2891, 0.4929, 0.8289, 
                        1.3828, 2.2961, 3.8019])
layer_nodes = np.array([0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 
                        1.0380, 1.7276, 2.8646])

# Get the mapping from component key (cokey) to map unit key (mukey)
mukey_cokey = get_mukey_cokey(gdb_path)


############################################################
# Tables of interest
# 
# Explanations of the individual table columns' names are in:
# - SSURGO-Metadata-Table-Column-Descriptions-Report.pdf 
#   https://www.nrcs.usda.gov/sites/default/files/2022-08/SSURGO-Metadata-Table-Column-Descriptions-Report.pdf
#
# Explanations of the table columns' units are in: 
# - SSURGO-Metadata-Tables-and-Columns-Report.pdf
#   https://www.nrcs.usda.gov/sites/default/files/2022-08/SSURGO-Metadata-Tables-and-Columns-Report.pdf
#
# The relevant tables and columns are: 
# 
# chorizon
# - sand (% weight): sandtotal_r
# - silt (% weight): silttotal_r
# - clay (% weight): claytotal_r
# - organic matter (% weight): om_r
# - hydraulic conductivity (um/s): ksat_r
# - the distance from the top of the soil to the 
#   upper boundary of the soil horizon (cm): hzdept_r
# - .............................................
#   lower boundary ................... (cm): hzdepb_r
#
# chfrags
# - gravel (% volume): fragvol_r
#
# This table is already mukey level. No need to do it separately. 
# muaggatt
# - soil depth (cm): brockdepmin
############################################################

#-----------------------------------------------------------
# Collect the chorizon data variables
#-----------------------------------------------------------
var_list = ['sandtotal_r', 'silttotal_r', 'claytotal_r', 'om_r', 'ksat_r']
for var in var_list:
    # ---------------------------------------------------------------------------
    # Collect all the data for this variable
    # ---------------------------------------------------------------------------
    # ~5 min per variable
    variable_data = []
    src = fiona.open(gdb_path, layer = 'chorizon')
    print(len(src))
    for i, feat in tqdm( enumerate(src) ):
        vardata = feat['properties'][var]
        if vardata is None:
            continue
        cokey = feat['properties']['cokey']
        mukey = np.int32(mukey_cokey.loc[cokey, 'mukey'])
        copct = mukey_cokey.loc[cokey, 'comppct_r']
        dtop = feat['properties']['hzdept_r']
        dbot = feat['properties']['hzdepb_r']
        variable_data.append([cokey, mukey, copct, dtop, dbot, vardata])
    variable_data = pd.DataFrame(variable_data, columns = ['cokey', 'mukey', 'copct', 
                                                           'hzdept_r', 'hzdepb_r', var])
    variable_data = variable_data.set_index(['mukey', 'cokey']).sort_index()

    # ---------------------------------------------------------------------------
    # Interpolate the cokey level data to ELM depths, and weight-average to
    # mukey level
    # ~ 0.5h per variable => 2.5 h in total
    # ---------------------------------------------------------------------------
    write_mukey_hdf5(var, variable_data, layer_nodes, layer_depth, dst_dir)


#-----------------------------------------------------------
# Collect the chfrags table variables
# 'chkey' => chorizon table's variable
#-----------------------------------------------------------
var = 'fragvol_r'

# direct read is relatively fast (~5 min)
frag_data = gpd.read_file(gdb_path, layer='chfrags')
# sum up all the fragments sharing the same chkey
frag_data = frag_data[['chkey',var]].groupby('chkey').sum()

# match this with the soil depth in chorizon data
variable_data = []
src = fiona.open(gdb_path, layer = 'chorizon')
print(len(src))
for i, feat in tqdm( enumerate(src) ):
    chkey = feat['properties']['chkey']
    if chkey in frag_data.index:
        cokey = feat['properties']['cokey']
        mukey = np.int32(mukey_cokey.loc[cokey, 'mukey'])
        copct = mukey_cokey.loc[cokey, 'comppct_r']
        dtop = feat['properties']['hzdept_r']
        dbot = feat['properties']['hzdepb_r']
        vardata = frag_data.loc[chkey, 'fragvol_r']
        variable_data.append([cokey, mukey, copct, dtop, dbot, vardata])
variable_data = pd.DataFrame(variable_data, columns = ['cokey', 'mukey', 'copct', 
                                                       'hzdept_r', 'hzdepb_r', var])
variable_data = variable_data.set_index(['mukey', 'cokey']).sort_index()


# ---------------------------------------------------------------------------
# Interpolate the cokey level data to ELM depths, and weight-average to
# mukey level
# ~ 0.5h per variable
# ---------------------------------------------------------------------------
write_mukey_hdf5(var, variable_data, layer_nodes, layer_depth, dst_dir)
