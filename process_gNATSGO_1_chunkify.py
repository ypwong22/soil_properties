"""
The mapunit key geotiff of the gNATSGO database is too large. Divide it up 
into smaller tiles for easier processing. 

Original shape: (96751, 153996)
"""
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from constants import path_data, path_intrim

row_splits = [16125] * 5 + [16126] # Sum = 96751
col_splits = [12833] * 12  # Sum = 153996

src_file = os.path.join(path_data, 'FY2024_gNATSGO_mukey_grid', 'FY2024_gNATSGO_mukey_grid.tif')
dst_dir = os.path.join(path_intrim, 'map_predictors', 'gNATSGO')

# Div up the mukey file
src = rio.open(src_file)

if src.height != 96751 or src.width != 153996:
    raise ValueError(f'Input dimensions {src.height}x{src.width} don\'t match expected 96751x153996')

meta = src.meta.copy()
transform = src.transform

row_start = 0
piece_num = 0

for row_size in row_splits:
    col_start = 0

    for col_size in col_splits:
        meta.update({
            'height': row_size,
            'width': col_size,
            'transform': rio.windows.transform(
                rio.windows.Window(col_start, row_start, col_size, row_size),
                transform)
        })

        window = rio.windows.Window(col_start, row_start, col_size, row_size)
        data = src.read(window=window)

        dst_file = os.path.join(dst_dir, f'mukey_grid_{piece_num+1}.tif')
        dst = rio.open(dst_file, 'w', **meta)
        dst.write(data)
        dst.close()

        print(f'Wrote piece {piece_num + 1} with dimensions {row_size}x{col_size}')

        col_start += col_size
        piece_num += 1

    row_start += row_size

src.close()


# Examine the results
def check_output(dst_file, figsize=(15, 10), dpi=600):
    """ Plotting function to examine the output is correct """

    us_states = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    us_states = us_states[us_states['continent'] == 'North America']

    with rio.open(dst_file) as src:
        dst_crs = 'EPSG:4326'

        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        dst_array = np.zeros((height, width), dtype=np.float32)

        reproject(
            source=rio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        if src.nodata is not None:
            dst_array = np.ma.masked_where(dst_array == src.nodata, dst_array)

    fig, ax = plt.subplots(figsize=figsize)

    us_states.to_crs('epsg:4326').plot(ax=ax, alpha=0.5, color='none', edgecolor='r')

    img = ax.imshow(dst_array, 
                    extent=[transform[2], transform[2] + transform[0] * width,
                            transform[5] + transform[4] * height, transform[5]],
                    cmap='viridis',
                    alpha=0.7)

    ax.set_xlim([-125, -65])
    ax.set_ylim([25, 50])
    ax.set_title(dst_file.split('/')[-1])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.savefig(dst_file.replace('tif','png'), dpi=dpi, bbox_inches='tight')
    plt.close()

for i in range(72):
    dst_file = os.path.join(dst_dir, f'mukey_grid_{i+1}.tif')
    check_output(dst_file)