""" Create global HWSD2 geotif files """
import rasterio as rio
from rasterio.transform import from_origin
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


path_hwsd = os.path.join(os.environ['PROJDIR'], 'Soil_Properties', 'data', 'HWSD2')
path_out = os.path.join(os.environ['PROJDIR'], 'Soil_Properties', 'intermediate', 'HWSD2')

layer = REPLACE # {1..8}
layer_ = f'D{layer}'


# (1) Read map file
src  = rio.open(os.path.join(path_hwsd, 'HWSD2.bil'))
band1 = src.read(1)


# (2) Subset to CONUS

# Get lat & lon for the rectangular grid
width, height = src.width, src.height

lon = np.full(width, np.nan)
lat = np.full(height, np.nan)

for row in range(height):
    x, y = src.xy(row, 0)
    lat[row] = y

for col in range(width):
    x, y = src.xy(0, col)
    lon[col] = x


lat_bounds = [23.0,54.5]
lon_bounds = [-125.5,-66.5]
band1 = band1[(lat >= lat_bounds[0]) & (lat <= lat_bounds[1]), 
              :][:, (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])]
lon = lon[(lon >= lon_bounds[0]) & (lon <= lon_bounds[1])]
lat = lat[(lat >= lat_bounds[0]) & (lat <= lat_bounds[1])]
print(band1.shape)


# (3) Obtain the map unit keys, excluding no data
mu_list = list(np.unique(band1.reshape(-1)))
mu_list.remove(65535)

# (4) Obtain soil property values from the MS Access Database
soildata = pd.read_csv(os.path.join(path_hwsd, 'HWSD2_LAYERS.csv'))

# List the soil layer depth
# --- only the 0-20cm and 20-40cm are relevant
layer_list = []
for layer in [f'D{i}' for i in range(1, 8)]:
    filt = soildata['LAYER'] == layer
    topdepth = soildata.loc[filt, 'TOPDEP'].iloc[0]
    botdepth = soildata.loc[filt, 'BOTDEP'].iloc[0]
    print(layer, topdepth, botdepth)
    layer_list.append(f'{topdepth}-{botdepth}cm')
print(layer_list)

# for each variable and layer, save to file
#
# Relevant variables are
# 
# COARSE    % volume    coarse fragments
# SAND      % weight    sand
# SILT      % weight    silt
# CLAY      % weight    clay
# BULK      g/cm3       bulk density
# ORG_CARBON    % weight    organic carbon content

for var in ['COARSE', 'SAND', 'SILT', 'CLAY', 'BULK', 'ORG_CARBON']:
    # for layer in [f'D{i}' for i in range(1, 8)]:
    #    print(var, layer)
    print(var,layer)

    band_temp = np.full(band1.shape, np.nan)
    for mu in tqdm(mu_list):
        value = soildata.query('LAYER == @layer_ & HWSD2_SMU_ID == @mu')[var]
        share = soildata.query('LAYER == @layer_ & HWSD2_SMU_ID == @mu')['SHARE']
        value = np.sum(value.values * share.values) / 100.
        band_temp = np.where(band1 == mu, value, band_temp)
    band_temp = np.where(band1 == int(src.nodata), 1e20, band_temp)

    profile = dict(src.profile)

    transform = from_origin(lon[0] + 0.5*(lon[0]-lon[1]), lat[0] + 0.5*(lat[0]-lat[1]),
                            lon[1]-lon[0], 
                            lat[0]-lat[1])  # (lon_min, lat_max, pixel_width, pixel_height)
    crs = rio.crs.CRS.from_epsg(4326)

    profile['driver'] = 'GTiff'
    profile['width'] = band_temp.shape[1]
    profile['height'] = band_temp.shape[0]
    profile['dtype'] = np.float64
    profile['nodata'] = 1e20
    profile['crs'] = crs
    profile['transform'] = transform
    file_out = os.path.join(path_out, f'{var}_{layer}.tif')
    dst = rio.open(file_out, 'w', **profile)
    dst.write(band_temp, 1)
    dst.close()

src.close()


# (5) Calculate the porosity
BD_om = 1.3
BD_minerals = 2.71
BD_gravels = 2.80
vf_pores_gravels = 0.24


#for layer in [f'D{i}' for i in range(1, 8)]:
data_list = {}
for var in ['COARSE', 'SAND', 'SILT', 'CLAY', 'BULK', 'ORG_CARBON']:
    h = rio.open(os.path.join(path_out, f'{var}_{layer}.tif'))
    data_list[var] = h.read(1, masked = True)
    data_list[var].mask = np.logical_or(data_list[var].mask, (data_list[var].data < 0))
    profile = dict(h.profile)
    h.close()

# volumetric fraction
vf_gravels_s = data_list['COARSE'] * (1 - vf_pores_gravels) / 100
# weight fraction of SOM within fine earth
wf_om_fine_earth = data_list['ORG_CARBON'] * 1.724 / 100
# volume fraction of SOM within fine earth
vf_om_fine_earth = np.ma.minimum(wf_om_fine_earth * data_list['BULK'] / BD_om, 1)

# Bulk density of soil (GRAVELS + MINERALS + ORGANIC MATTER)
# BD is the BULK DENSITY OF FINE EARTH (MINERALS + ORGANIC MATTER)
BD_avg = (1 - vf_gravels_s/(1 - vf_pores_gravels)) * data_list['BULK'] + vf_gravels_s * BD_gravels

# Mass fraction of gravels
wf_gravels_s = vf_gravels_s * BD_gravels / BD_avg
wf_sand_s = data_list['SAND'] / 100 * (1 - wf_om_fine_earth) * (1 - wf_gravels_s)
wf_silt_s = data_list['SILT'] / 100 * (1 - wf_om_fine_earth) * (1 - wf_gravels_s)
wf_clay_s = data_list['CLAY'] / 100 * (1 - wf_om_fine_earth) * (1 - wf_gravels_s)
wf_om_s = wf_om_fine_earth * (1 - wf_gravels_s)

# Volumetric fraction of soil constituents
vf_sand_s = wf_sand_s * BD_avg / BD_minerals
vf_silt_s = wf_silt_s * BD_avg / BD_minerals
vf_clay_s = wf_clay_s * BD_avg / BD_minerals
vf_om_s = wf_om_s * BD_avg / BD_om

# Particle density of soil (minerals + organic matter + gravels)
BD_particle_inverse = wf_gravels_s / BD_gravels + \
    (1 - wf_gravels_s) * ((1 - wf_om_fine_earth)/BD_minerals + wf_om_fine_earth/BD_om)
BD_particle = 1 / BD_particle_inverse

# Porosity of soil
vf_pores_s = 1 - BD_avg * BD_particle_inverse

# Save to file
profile['dtype'] = np.float32
profile['nodata'] = 1e20
profile['driver'] = 'GTiff'
profile['height'] = vf_pores_s.shape[0]
profile['dtype'] = np.float32
profile['nodata'] = 1e20
file_out = os.path.join(path_out, f'porosity_{layer}.tif')
dst = rio.open(file_out, 'w', **profile)
dst.write(vf_pores_s, 1)
dst.close()
