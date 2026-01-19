"""
Merge the available soil properties data into 1km ELM surfdata input

pH          soil pH
PCT_CLAY    percent clay                        %
PCT_SAND    percent sand                        %
ORGANIC     organic content                     kg/m3
cfvo        coarse fragments                    % volume
bd_col      bulk density                        kg/m3
aveDTB      depth to bedrock                    m below surface

All from SoilGrids, except:
- aveDTB: Pelletier et al. 2016 (rounded off to the nearest meter)
- aveDTB2: SoilGrids (too deep)
"""
from netCDF4 import Dataset
import os
import rasterio as rio
from constants import *
from utils import vert_interp
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import subprocess
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def vert_interp_elm(data):
    """ helper function to interpolate SoilGrids data to
        [layer, lat, lon] into elm layers. The two are on the same vertical grids.
    """
    orig_shape = data.shape
    # Reshape into [points, layer] to facilitate interpolation
    data_in = data.reshape(orig_shape[0], -1).T
    
    filt = np.where(~np.isnan(data_in[:,0]))[0]
    data_valid = data_in[filt, :]

    # Temporary reshape to interpolate to ELM layers
    input_interfaces = np.array([0, 5, 15, 30, 60, 100, 200])
    input_nodes = np.convolve(input_interfaces, [0.5, 0.5], mode = 'valid')
    output_data_ = vert_interp(elm_nodes, input_nodes, data_valid, False, elm_interface, input_interfaces)

    # Reshape back 
    output_data = np.full([data_in.shape[0], len(elm_nodes)], np.nan)
    output_data[filt, :] = output_data_
    output_data = output_data.T
    
    if len(orig_shape) == 3:
        output_data = output_data.reshape(len(elm_nodes), orig_shape[1], orig_shape[2])

    return output_data


def get_nn_indices(ref_axis, target_axis):
    """
    Find the nearest neighbor indices in ref_axis
    to the values in target_axis
    """
    if ref_axis[1] < ref_axis[0]: # Descending
        sorter = np.arange(len(ref_axis))[::-1]
        ref_sorted = ref_axis[sorter]
    else: # Ascending
        sorter = np.arange(len(ref_axis))
        ref_sorted = ref_axis

    # each value in the target_axis, i.e. target_axis[i],
    # can be inserted between ref_sorted[idx[i]-1] and
    #                         ref_sorted[idx[i]]
    # to maintain the sorted ascending order
    idx = np.searchsorted(ref_sorted, target_axis)
    idx = np.clip(idx, 0, len(ref_sorted) - 1)

    idx_left = np.clip(idx - 1, 0, len(ref_sorted) - 1)
    dist_right = np.abs(target_axis - ref_sorted[idx])
    dist_left = np.abs(target_axis - ref_sorted[idx_left])
    
    use_left = dist_left < dist_right
    idx[use_left] = idx_left[use_left]
    
    return sorter[idx]


def get_tif_coords(filename):
    """ Get 1D lat and lon vectors from a GeoTIFF """
    with rio.open(filename) as src:
        height, width = src.height, src.width
        transform = src.transform
    
    lons, _ = rio.transform.xy(transform, [0]*width, np.arange(width), offset='center')
    _, lats = rio.transform.xy(transform, np.arange(height), [0]*height, offset='center')
    
    return np.array(lats), np.array(lons)


def read_soilgrids(varname, target_lat, target_lon):
    """ helper to read a SoilGrids geotiff 
    https://gee-community-catalog.org/projects/isric/#citation
    """
    # Map varname to filename
    if varname == 'pH':
        ff = 'phh2o'
    elif varname == 'PCT_CLAY':
        ff = 'clay'
    elif varname == 'PCT_SAND':
        ff = 'sand'
    elif varname == 'ORGANIC':
        ff = 'soc'
    elif varname == 'cfvo':
        ff = 'cfvo'
    elif varname == 'bd_col':
        ff = 'bdod'
    else:
        raise Exception('Not implemented')

    # Get source coordinates from the first layer file
    filename_ref = os.path.join(path_data, 'SoilGrids', f'na_{ff}_0-5cm.tif')
    src_lat, src_lon = get_tif_coords(filename_ref)

    # Find nearest neighbor indices
    ilat = get_nn_indices(src_lat, target_lat)
    ilon = get_nn_indices(src_lon, target_lon)
    
    # Optimization: Read only the bounding box
    lat_min, lat_max = ilat.min(), ilat.max()
    lon_min, lon_max = ilon.min(), ilon.max()

    data = np.full([6, len(target_lat)], np.nan)
    for i, layer in enumerate(['0-5cm','5-15cm','15-30cm','30-60cm','60-100cm','100-200cm']):
        filename = os.path.join(path_data, 'SoilGrids', f'na_{ff}_{layer}.tif')
        with rio.open(filename) as h:
            window = rio.windows.Window(lon_min, lat_min, 
                                      lon_max - lon_min + 1, 
                                      lat_max - lat_min + 1)
            temp = h.read(1, window=window, masked=True)
            
            # Extract points using local indices
            vals = temp[ilat - lat_min, ilon - lon_min]
            data[i,:] = np.where(np.ma.getmaskarray(vals), np.nan, vals)

            if varname in ['pH', 'cfvo','PCT_CLAY','PCT_SAND','ORGANIC']:
                # the original GEE data is scaled up by 10
                data[i,:] = data[i,:] / 10
            elif varname == 'bd_col':
                # the original GEE data is scaled up by 100
                data[i,:] = data[i,:] / 100 * 1e3 # kg/dm3 => kg/m3

    # Vertical interpolation to ELM layers
    data = vert_interp_elm(data)

    if varname == 'ORGANIC':
        # g/kg * dry soil bulk density = kg/m3
        bd = read_soilgrids('bd_col', target_lat, target_lon)
        cfvo = read_soilgrids('cfvo', target_lat, target_lon)
        data = data / 1e3 * bd * (1 - cfvo/100)

    return data


def read_1km(varname, target_lat, target_lon):
    """ helper to read 1 variable from source file
    """
    path_src = os.path.join(os.environ['SHARDIR'], 'Soil_Properties', 'data', 'global_cf_float')
    if varname in ['PCT_CLAY', 'PCT_SAND', 'ORGANIC']:
        filename = f'{varname}_10layer_1k_c230606.nc'
    else:
        filename = f'{varname}_1k_c230606.nc'
    
    with Dataset(os.path.join(path_src, filename), mode="r", format="NETCDF4") as ds:
        lat = ds['lat'][:]
        lon = ds['lon'][:]

        ilat = get_nn_indices(lat, target_lat)
        ilon = get_nn_indices(lon, target_lon)

        # Read data using the indices
        # Optimization: read only the bounding box
        lat_min, lat_max = ilat.min(), ilat.max()
        lon_min, lon_max = ilon.min(), ilon.max()

        lat_slice = slice(lat_min, lat_max + 1)
        lon_slice = slice(lon_min, lon_max + 1)

        # dimension: (lat, lon) or (layer, lat, lon)
        if ds[varname].ndim == 3:
            data_sub = ds[varname][:, lat_slice, lon_slice]
            data = data_sub[:, ilat - lat_min, ilon - lon_min]
        else:
            data_sub = ds[varname][lat_slice, lon_slice]
            data = data_sub[ilat - lat_min, ilon - lon_min]

    return data


def append_layer(data, var_name, fill_value=-9999.0, **attrs):
    """ helper to append a new variable to the 1D unstructured NetCDF file """
    nc_path = os.path.join(os.environ['SHARDIR'], '..', 'wangd', 'kiloCraft',
                           'TES_cases_data', 'Daymet_ERA5_TESSFA_NORTH', 'entire_domain', 'domain_surfdata',
                           'NORTHERA5_surfdata.TES_NORTHERA5.4km.1d.c251009.nc_yw20260115')

    with Dataset(nc_path, mode="a") as ds:
        if var_name not in ds.variables:
            if data.ndim == 2: # (layer, gridcell)
                dims = ("nlevsoi", 'gridcell')
            else: # (gridcell)
                dims = ('gridcell',)
                
            v = ds.createVariable(
                    var_name, np.float32,
                    dims,
                    zlib=True, complevel=4,
                    fill_value = fill_value
            )
            v.missing_value = fill_value
            for k, v_attr in attrs.items():
                setattr(v, k, v_attr)
        
        v = ds.variables[var_name]
        v[:] = np.ma.masked_invalid(data)


def check_plot(varname, data_new, lat_new, lon_new):
    """ Verify the interpolation by plotting original vs new data """
    print(f"Plotting verification for {varname}...")
    
    # Handle layers: ensure data_new is (nlayer, npoints)
    if data_new.ndim == 1:
        data_new = data_new.reshape(1, -1)
    
    nlayer = data_new.shape[0]
    
    # Harmonize range
    vmin = np.nanmin(data_new)
    vmax = np.nanmax(data_new)

    # Setup plot
    # squeeze=False ensures ax is always (nlayer, 2)
    fig, ax = plt.subplots(nlayer, 2, figsize=(12, 4*nlayer), 
                           subplot_kw={'projection': ccrs.PlateCarree()},
                           squeeze=False)
    
    # 1. Plot New Data (Scatter)
    for n in range(nlayer):
        val_new = data_new[n, :]
        
        sc = ax[n,1].scatter(lon_new, lat_new, c=val_new, s=1, cmap='viridis', 
                             transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
        ax[n,1].set_title(f"New 4km Data (Layer {n+1})")
        ax[n,1].coastlines()
        plt.colorbar(sc, ax=ax[n,1], fraction=0.046, pad=0.04)

    # 2. Plot Original Data
    try:
        path_src_1km = os.path.join(path_data, 'global_cf_float')
        
        if varname in ['ELEVATION', 'ORGANIC', 'PCT_CLAY', 'PCT_SAND', 'SLOPE', 'STDEV_ELEV']:
            # 1km NetCDF logic
            fname = f'{varname}_10layer_1k_c230606.nc' if varname in ['PCT_CLAY', 'PCT_SAND', 'ORGANIC'] else f'{varname}_1k_c230606.nc'
            fpath = os.path.join(path_src_1km, fname)
            
            with Dataset(fpath) as ds:
                # Read a subset roughly covering the domain
                lat = ds['lat'][:]
                lon = ds['lon'][:]
                
                # Find indices (approximate)
                lat_mask = (lat >= lat_new.min() - 0.1) & (lat <= lat_new.max() + 0.1)
                lon_mask = (lon >= lon_new.min() - 0.1) & (lon <= lon_new.max() + 0.1)

                extent = [lon[lon_mask].min(), lon[lon_mask].max(), lat[lat_mask].min(), lat[lat_mask].max()]

                # Check dimensions of source variable
                src_var = ds[varname]
                
                for n in range(nlayer):
                    if src_var.ndim == 3:
                        # (layer, lat, lon)
                        if n < src_var.shape[0]:
                            data_orig = src_var[n, lat_mask, :][:, lon_mask]
                        else:
                            continue 
                    else:
                        # (lat, lon)
                        if n == 0:
                            data_orig = src_var[lat_mask, :][:, lon_mask]
                        else:
                            continue

                    im = ax[n,0].imshow(data_orig, extent=extent, origin='lower', cmap='viridis', 
                                        transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
                    ax[n,0].set_title(f"Original 1km ({fname}) Layer {n+1}")
                    plt.colorbar(im, ax=ax[n,0], fraction=0.046, pad=0.04)
                    ax[n,0].coastlines()

        elif varname == 'pH':
            sg_layers = ['0-5cm','5-15cm','15-30cm','30-60cm','60-100cm','100-200cm']
            for n in range(nlayer):
                if n < len(sg_layers):
                    layer = sg_layers[n]
                    fpath = os.path.join(path_data, 'SoilGrids', f'na_phh2o_{layer}.tif')
                    if os.path.exists(fpath):
                        with rio.open(fpath) as src:
                            window = rio.windows.from_bounds(lon_new.min(), lat_new.min(), 
                                                           lon_new.max(), lat_new.max(), src.transform)
                            data_orig = src.read(1, window=window, masked=True) / 10.0
                            bounds = src.window_bounds(window)
                            extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
                            im = ax[n,0].imshow(data_orig, extent=extent, origin='upper', cmap='viridis', 
                                                transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
                            ax[n,0].set_title(f"Original 1km (SoilGrids pH {layer})")
                            plt.colorbar(im, ax=ax[n,0], fraction=0.046, pad=0.04)
                            ax[n,0].coastlines()
                else:
                    ax[n,0].text(0.5, 0.5, "No corresponding source layer", 
                                 transform=ax[n,0].transAxes, ha='center')

    except Exception as e:
        print(f"Could not plot original data for {varname}: {e}")

    plt.savefig(os.path.join(os.environ['SHARDIR'], 'Soil_Properties', 'intermediate', 'merged_surfdat', 
                             f'check_{varname}.png'), dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    """ Overwrite the netCDF4 file that will be the repository of data variables """
    nc_path = os.path.join(os.environ['SHARDIR'], '..', 'wangd', 'kiloCraft',
                           'TES_cases_data', 'Daymet_ERA5_TESSFA_NORTH', 'entire_domain', 'domain_surfdata',
                           'NORTHERA5_surfdata.TES_NORTHERA5.4km.1d.c251009.nc_yw20260115')
    with Dataset(nc_path, mode="a", format="NETCDF4") as ds: 
        # ---- dimensions --------------------------------------------------------
        lat = ds['lat'][:]
        lon = ds['lon'][:]
        gridlat = np.array([lat[y] for y in ds['gridYID']])
        gridlon = np.array([lon[x] for x in ds['gridXID']])

        # 'ASPECT', 'CANOPY_HEIGHT_BOT', 'CANOPY_HEIGHT_TOP', 'SKY_VIEW_FACTOR', 'TERRAIN_CONFIG'
        for var in ['ELEVATION', 'ORGANIC', 'PCT_CLAY', 'PCT_SAND', 'SLOPE', 'STDEV_ELEV']:
            if var == 'STDEV_ELEV':
                var_ = 'STD_ELEV'
            elif var == 'ELEVATION':
                var_ = 'TOPO'
            else:
                var_ = var

            data = read_1km(var, gridlat, gridlon)
            check_plot(var, data, gridlat, gridlon)
            if len(data.shape) == 2:
                ds[var_][:,:] = data
            else:
                ds[var_][:] = data
            ds.sync()

    data = read_soilgrids('pH', gridlat, gridlon)
    check_plot('pH', data, gridlat, gridlon)
    append_layer(data, 'pH', -9999.0, 
                 long_name = 'Soil pH', units = '')
    