""" Convert the NetCDF outputs of process gNATSGO_4_interp.sh to CONUS netcdf like Lingcheng Li's 
    1km data using nearest neighbor mapping, since both are ~1km.
    Cut out Lingcheng Li's 1km soil texture data to the same extent. 

    Still need to do HWSD2
"""
from netCDF4 import Dataset
import xarray as xr
import os
from constants import path_data, path_intrim
import numpy as np


def target_coords(var):
    """ Obtain the target coordinates from Lingcheng Li's file """
    nc = Dataset(os.path.join(path_data, 'global_cf_float', f'PCT_{var}_10layer_1k_c230606.nc'))
    lon_target = nc['lon'][:]
    lat_target = nc['lat'][:]
    nc.close()
    return lon_target, lat_target


def match_coords(src, target):
    """ For each value in src array, find the index of the nearest neighbor in the target array. 
        Return the array of this index, and the array that consists of the nearest neighbors,
            sorted in the order of the src array
    """
    # find the distance between 
    dist = np.abs(np.broadcast_to(src.reshape(-1,1), [len(src), len(target)]) - \
                  np.broadcast_to(target.reshape(1,-1), [len(src), len(target)]))
    index = np.argmin(dist, axis = 1)
    neighbors = target[index]
    return index, neighbors


def subset_target(var, ind_lat, ind_lon):
    """ Subset Lingcheng Li's file to the same area as gNATSGO """
    ds = xr.open_dataset(os.path.join(path_data, 'global_cf_float',
                                      f'PCT_{var}_10layer_1k_c230606.nc'))

    # Note: isel is used for integer-based indexing
    ds_subset = ds.isel(lat=ind_lat, lon=ind_lon)

    ds_subset[f'PCT_{var_out}'].encoding = {'_FillValue': -9999.}
    for vv in ['lat', 'lon', 'layer', 'layer_depth']:
        ds_subset[vv].encoding = {'_FillValue': None}

    ds_subset.to_netcdf(os.path.join(path_intrim, f'PCT_{var}_10layer_1k_c230606.nc_CONUS'), 
                        format='NETCDF4')

    ds.close()
    ds_subset.close()


def gNATSGO_data(var_in, target_lon, target_lat):
    nlayers = 10
    layer_depths = np.array([0.0175, 0.0451, 0.0906, 0.1655, 0.2891, 0.4929, 
                             0.8289, 1.3828, 2.2961, 3.8019])

    # collect the data
    for band in range(1,nlayers + 1):
        nc = Dataset(os.path.join(path_data, 'gNATSGO_CONUS', 'mukey_divide', 
                                  f'{var_in}_{band}.nc'))
        if band == 1:
            nx = len(nc['lon'][:])
            ny = len(nc['lat'][:])
            lon = nc['lon'][:]
            lat = nc['lat'][:]
            data_collect = np.empty([nlayers, ny, nx])
        data_collect[band-1, :, :] = nc['Band1'][:, :]
        nc.close()

    # find the nearest neighbors in Lingcheng Li's data
    ind_lat, new_lat = match_coords(lat, target_lat)
    ind_lon, new_lon = match_coords(lon, target_lon)

    # create the data variables
    layer = xr.DataArray(data=np.arange(1, nlayers+1), dims=['layer'],
                         attrs={'standard_name': 'soil_layer', 
                                'long_name': 'soil layer',
                                'units': 'layer'})
    layer_depth = xr.DataArray(data = layer_depths,
                               dims=['layer'],
                               attrs={'standard_name': 'soil_layer_bottom_depth',
                                      'long_name': 'the bottom depth for each soil layer',
                                      'units': 'm'})
    lat = xr.DataArray(data=new_lat, dims=['lat'],
                       attrs={'standard_name': 'latitude',
                              'long_name': 'latitude',
                              'units': 'degrees_north'})
    lon = xr.DataArray(data=new_lon, dims=['lon'],
                       attrs={'standard_name': 'longitude',
                              'long_name': 'longitude',
                              'units': 'degrees_east'})
    var_out_lower = var_out.lower()
    data_collect = xr.DataArray(data = data_collect, 
                                dims = ['layer', 'lat', 'lon'], 
                                coords = {'layer': layer, 'lat': lat, 'lon': lon}, 
                                attrs = {'standard_name': f'percent_{var_out_lower}',
                                        'long_name': f'mass fraction of {var_out_lower} in soil',
                                        'units': '%',
                                        'nlevsoi': '10 soil layer bottom depth (m), 0.0175, 0.0451, 0.0906, 0.1655, 0.2891, 0.4929, 0.8289, 1.3828, 2.2961, 3.8019. For the use of E3SM Land Model'
                                        })


    # Combine all DataArrays into a Dataset
    ds = xr.Dataset({
        f'PCT_{var_out}': data_collect,
        'lat': lat,
        'layer': layer,
        'layer_depth': layer_depth,
        'lon': lon
    })

    ds[f'PCT_{var_out}'].encoding = {'_FillValue': -9999.}
    for vv in ['lat', 'lon', 'layer', 'layer_depth']:
        ds[vv].encoding = {'_FillValue': None}

    # Save to NetCDF file
    ds.to_netcdf(os.path.join(path_intrim, f'PCT_{var_out}_gNATSGO.nc'), format='NETCDF4')

    return ind_lat, ind_lon


if __name__ == '__main__':
    var_in = 'sandtotal_r'
    var_out = 'SAND'
    lon_target, lat_target = target_coords(var_out)
    ind_lat, ind_lon = gNATSGO_data(var_in, lon_target, lat_target)
    subset_target(var_out, ind_lat, ind_lon)