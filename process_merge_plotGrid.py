import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Step 1: Read the NetCDF file
filename = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/kiloCraft/TES_cases_data/ACCESS_TESSFA_North/entire_domain/domain_surfdata/domain.lnd.TES_NORTHACCESS.4km.1d.c250521.nc'  # Replace with your file path if needed
dataset = nc.Dataset(filename)

# Step 2: Get the coordinates and gridID from the dataset
x_coords = dataset.variables['xc'][:]  # x coordinate values
y_coords = dataset.variables['yc'][:]  # y coordinate values
mask = dataset.variables['mask'][:] # 1 means land

filt = mask > 0

fig, ax = plt.subplots(figsize = (20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines()
ax.scatter(x_coords[filt], y_coords[filt], mask[filt])
plt.savefig('2D_map.png', dpi = 600., bbox_inches = 'tight')

# Close the dataset
dataset.close()
