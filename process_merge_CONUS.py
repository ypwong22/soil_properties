"""
Merge the available soil properties data into 1km input ELM friendly format.

pH          soil pH
PCT_CLAY    percent clay                        %
PCT_SAND    percent sand                        %
ORGANIC     organic content                     kg/m3
cfvo        coarse fragments                    % volume
bd_col      bulk density                        kg/m3
hksat_col   saturated hydraulic conductivity    mm H2O/s
watsat_col  saturated water content             m3/m3
aveDTB      depth to bedrock                    m below surface

Among those, the sources are

- pH, PCT_CLAY, PCT_SAND, bd_col, hksat_col, watsat_col: POLARIS
- ORGANIC: POLARIS, with cfvo from SoilGrids (gNATSGO is less complete and does not have bd_col)
- ORGANIC2: SoilGrids (processed by Li et al. 2024)
- ORGANIC3: gNATSGO, with bd_col from POLARIS and cfvo from SoilGrids
- cfvo: SoilGrids, the only available source
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


def get_latlon(tif):
    """ Get the lat & lon vectors """
    with rio.open(tif) as src:
        h, w         = src.height, src.width      # 2913 × 6694
        transform    = src.transform              # Affine from the profile
        crs          = src.crs                    # EPSG:4326  (lat/lon degrees)

    # --- build row / column grids -----------------------------------------------
    # rows = 0 … h-1 (southward is +row); cols = 0 … w-1 (eastward is +col)
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))

    # --- convert to geographic coordinates --------------------------------------
    lon, lat = rio.transform.xy(transform, rows, cols, zs = None, offset = 'center')
    lon = lon.reshape(cols.shape)[0,:]
    lat = lat.reshape(rows.shape)[:,0]

    return lat, lon


def create_file(lat, lon):
    """ Create the netCDF4 file that will be the repository of data variables """
    nc_path = os.path.join(path_intrim, 'merged_properties_conus.nc')

    with Dataset(nc_path, mode="w", format="NETCDF4") as ds: 
        # ---- dimensions --------------------------------------------------------
        ds.createDimension("lat",  len(lat))      # fixed size
        ds.createDimension("lon",  len(lon))      # fixed size
        ds.createDimension("layer", 10)           # num soil layers in ELM

        # ---- coordinate variables ---------------------------------------------
        vlat = ds.createVariable("lat", np.float32, ("lat",))
        vlat.standard_name = "latitude"
        vlat.units         = "degrees_north"
        vlat[:] = lat

        vlon = ds.createVariable("lon", np.float32, ("lon",))
        vlon.standard_name = "longitude"
        vlon.units         = "degrees_east"
        vlon[:] = lon

        vsoi = ds.createVariable("layer", np.float32, ("layer",))
        vsoi.standard_name = "ELM soil layer"
        vsoi.units         = ""
        vsoi[:] = np.arange(1,11)

        # ---- global attrs (optional) ------------------------------------------
        ds.title      = "Soil properties data from multiple sources"
        ds.source     = \
            "- pH, PCT_CLAY, PCT_SAND, bd_col, hksat_col, watsat_col: POLARIS\n" + \
            "- ORGANIC: POLARIS, with cfvo from SoilGrids (gNATSGO is less complete and does not have bd_col)\n" + \
            "- ORGANIC2: SoilGrids (processed by Li et al. 2024)\n" + \
            "- ORGANIC3: gNATSGO, with bd_col from POLARIS and cfvo from SoilGrids\n" + \
            "- cfvo: SoilGrids, the only available source\n" + \
            "- aveDTB: Pelletier et al. 2016 (rounded off to the nearest meter)\n" + \
            "- aveDTB2: Global_Soil_Regolith_Sediment_1304 (too deep)\n\n" + \
            "Pelletier, J.D., P.D. Broxton, P. Hazenberg, X. Zeng, P.A. Troch, G. Niu, Z.C. Williams, M.A. Brunke, and D. Gochis. 2016. Global 1-km Gridded Thickness of Soil, Regolith, and Sedimentary Deposit Layers. ORNL DAAC, Oak Ridge, Tennessee, USA. http://dx.doi.org/10.3334/ORNLDAAC/1304" + \
            "Li, L., Bisht, G., Hao, D., and Leung, L. R.: Global 1 km land surface parameters for kilometer-scale Earth system modeling, Earth Syst. Sci. Data, 16, 2007-2032, https://doi.org/10.5194/essd-16-2007-2024, 2024."
        ds.geodetic_datum = "EPSG:4326 (WGS84)"

    return


def append_layer(data, var_name, fill_value=-9999.0, **attrs):
    """ helper to append a new raster (3D NumPy array) as a fresh layer.

    Append a 3D (layer, lat, lon) array to the netCDF file
      - nc_file  : path to the NetCDF file created above
      - data     : numpy array with shape (len(lat), len(lon), len(layer))
      - var_name : variable that will store the layers
      - **attrs  : any keyword=value pairs you want stored as the data variable's attributes
                   e.g. long name, units, 
    """
    nc_path = os.path.join(path_intrim, 'merged_properties_conus.nc')

    with Dataset(nc_path, mode="a") as ds:
        if var_name not in ds.variables:
            if len(data.shape) == 3:
                v = ds.createVariable(
                        var_name, np.float32,
                        ("layer", "lat", "lon"),
                        zlib=True, complevel=4,  # compressed
                        fill_value = fill_value
                )
            else:
                v = ds.createVariable(
                        var_name, np.float32,
                        ("lat", "lon"),
                        zlib=True, complevel=4,  # compressed
                        fill_value = fill_value
                )
            v.missing_value = fill_value
            for k, v_attr in attrs.items():
                setattr(v, k, v_attr)
        v = ds.variables[var_name]
        if len(data.shape) == 3:
            v[:, :, :] = np.ma.masked_invalid(data)
        else:
            v[:, :] = np.ma.masked_invalid(data)

        ds.sync()

    return


def vert_interp_elm(data):
    """ helper function to interpolate POLARIS & SoilGrids data
        [layer, lat, lon] into elm layers. The two are on the same vertical grids.
    """
    # Reshape into [latxlon, layer] to facilitate interpolation
    data = data.reshape(data.shape[0], -1).T
    filt = np.where(~np.isnan(data[:,0]))[0]
    data = data[filt, :]

    # Temporary reshape to interpolate to ELM layers
    input_interfaces = np.array([0, 5, 15, 30, 60, 100, 200])
    input_nodes = np.convolve(input_interfaces, [0.5, 0.5], mode = 'valid')
    output_data_ = vert_interp(elm_nodes, input_nodes, data, False, elm_interface, input_interfaces)

    # Reshape back into [layer, lat, lon]
    output_data = np.full([len(lon)*len(lat), len(elm_nodes)], np.nan)
    output_data[filt, :] = output_data_
    output_data = output_data.T.reshape([len(elm_nodes), len(lat), len(lon)])

    return output_data



def read_cfvo(lat, lon):
    """ Read the coarse fragments (%) from SoilGrids because it is necessary to calculate organic matter content """

    # Spatial interpolation to POLARIS grid
    data = np.full([6, len(lat), len(lon)], np.nan)
    lon2d, lat2d = np.meshgrid(lon, lat)
    for i, layer in enumerate(['0-5cm','5-15cm','15-30cm','30-60cm','60-100cm','100-200cm']):
        filename = os.path.join(path_data, 'SoilGrids', f'cfvo_{layer}_mean.tif')
        lat_, lon_ = get_latlon(filename)
        with rio.open(filename) as h:
            temp = h.read(1, masked = True)
            # the original GEE data is scaled up by 10
            temp = np.where(temp.mask, np.nan, temp.data / 10)

        lon2d, lat2d = np.meshgrid(lon, lat)

        f = RegularGridInterpolator((lat_ ,lon_), temp, 
                                    method = 'nearest', bounds_error=False,
                                    fill_value = np.nan)
        data[i,:,:] = f(np.column_stack([lat2d.ravel(), lon2d.ravel()])).reshape(lat2d.shape)

    # Vertical interpolation to ELM layers
    output_data = vert_interp_elm(data)

    return output_data



def read_polaris(varname, cfvo = None):
    """ helper to read a polaris geotiff """

    if varname == 'ORGANIC':
        try:
            assert (not cfvo is None)
        except:
            raise ValueError('Must input coarse fragments for ORGANIC calculation')

    # Map varname to filename
    if varname == 'pH':
        filename = os.path.join(path_data, 'POLARIS', f'polaris_ph_1km.tif')
    elif varname == 'PCT_CLAY':
        filename = os.path.join(path_data, 'POLARIS', f'polaris_clay_1km.tif')
    elif varname == 'PCT_SAND':
        filename = os.path.join(path_data, 'POLARIS', f'polaris_sand_1km.tif')
    elif varname == 'bd_col':
        filename = os.path.join(path_data, 'POLARIS', f'polaris_bd_1km.tif')
    elif varname == 'ORGANIC':
        filename = os.path.join(path_data, 'POLARIS', f'polaris_om_1km.tif')
    elif varname == 'hksat_col':
        filename = os.path.join(path_data, 'POLARIS', f'polaris_ksat_1km.tif')        
    elif varname == 'watsat_col':
        filename = os.path.join(path_data, 'POLARIS', f'polaris_theta_s_1km.tif')        
    else:
        raise Exception('Not implemented')

    # Read the data array in [layer, lat, lon]
    with rio.open(filename) as h:
        data = h.read()

    output_data = vert_interp_elm(data)

    if varname == 'bd_col':
        # g/cm3 => kg/m3
        output_data *= 1e3
    elif varname == 'ORGANIC':
        # % weight * dry soil bulk density = kg/m3
        output_data = (output_data / 100) * read_polaris('bd_col') * (1 - cfvo/100)
    elif varname == 'hksat_col':
        # cm/hr => mm/s
        output_data *= (10/3600)

    return output_data


def read_organic2(lat, lon):
    """ Read SoilGrids organic matter from Lingcheng Li's 1km dataset

        Li, L., Bisht, G., Hao, D., and Leung, L. R.: Global 1 km land surface parameters for kilometer-scale Earth system modeling, Earth Syst. Sci. Data, 16, 2007-2032, https://doi.org/10.5194/essd-16-2007-2024, 2024.
    """
    with Dataset(os.path.join(path_data, 'global_cf_float', 'ORGANIC_10layer_1k_c230606.nc')) as nc:
        lat_ = nc['lat'][:].data
        filt_y = (lat_ >= np.min(lat)) & (lat_ <= np.max(lat))

        lon_ = nc['lon'][:].data
        filt_x = (lon_ >= lon[0]) & (lon_ <= lon[-1])

        lat_ = lat_[filt_y][::-1]
        lon_ = lon_[filt_x]

        organic = nc['ORGANIC'][:, filt_y, filt_x][:, ::-1, :]
        organic = np.where(organic.mask, np.nan, organic.data)

    lon2d, lat2d = np.meshgrid(lon, lat)
    data = np.full([10, len(lat), len(lon)], np.nan)
    for i in range(10):
        f = RegularGridInterpolator((lat_ ,lon_), organic[i,:,:], 
                                    method = 'nearest', bounds_error=False,
                                    fill_value = np.nan)
        data[i,:,:] = f(np.column_stack([lat2d.ravel(), lon2d.ravel()])).reshape(lat2d.shape)
    return data


def read_gNATSGO(varname, lat, lon, cfvo = None):
    """ Read gNATSGO soil organic matter netcdf and resample by nearest neighbor
        to the POLARIS 1km grid. 
    """
    if varname == 'ORGANIC':
        try:
            assert (not cfvo is None)
        except:
            raise ValueError('Must input coarse fragments for ORGANIC calculation')

    if varname == 'ORGANIC':
        ff = 'om'
    elif varname == 'cfvo':
        ff = 'fragvol'

    data = np.full([10, len(lat), len(lon)], np.nan)

    lon2d, lat2d = np.meshgrid(lon, lat)
    for i in range(1, 11):
        with Dataset(os.path.join(path_intrim, 'gNATSGO', f'{ff}_r_{i}.nc')) as nc:
            lat_old = nc['lat'][:].data
            lon_old = nc['lon'][:].data
            temp = np.where(nc['Band1'][:,:].mask, np.nan, nc['Band1'][:, :].data)

            f = RegularGridInterpolator((lat_old, lon_old), temp, 
                                        method = 'nearest', bounds_error=False,
                                        fill_value = np.nan)
            data[i-1, :, :] = f(np.column_stack([lat2d.ravel(), 
                                                 lon2d.ravel()])).reshape(lat2d.shape)
    
    if varname == 'ORGANIC':
        # convert from % to kg/m3
        
        ##cfvo = read_gNATSGO('cfvo', lat, lon)
        ##cfvo2 = read_cfvo(lat, lon)
        ##cfvo = np.where(~((cfvo >= 0) & (cfvo <= 100)), cfvo2, cfvo)

        data = data / 100 * read_polaris('bd_col') * (1 - cfvo/100)

    return data


def read_aveDTB(lat, lon):
    """ Read the soil thickness from two sources because they do not agree

    (1) discretized into 0, 1, 2, ..., 50 m

    Pelletier, J.D., P.D. Broxton, P. Hazenberg, X. Zeng, P.A. Troch, G. Niu, Z.C. Williams, M.A. Brunke, and D. Gochis. 2016. Global 1-km Gridded Thickness of Soil, Regolith, and Sedimentary Deposit Layers. ORNL DAAC, Oak Ridge, Tennessee, USA. http://dx.doi.org/10.3334/ORNLDAAC/1304 

    (2) Global data set from SoilGrids, probably too deep. Uncensored data (censored 
       (everything > 2m is cut off) data is not the same.)

    Shangguan, W., Hengl, T., de Jesus, J.M., Yuan, H. and Dai, Y., 2017. Mapping the global depth to bedrock for land surface modeling. Journal of Advances in Modeling Earth Systems: Accepted.

    Interpolate to POLARIS lat lon
    """
    data_interp = {}
    for i in range(2):
        if i == 0:
            src = os.path.join(path_data, 'Global_Soil_Regolith_Sediment_1304', 'data', 
                                    'average_soil_and_sedimentary-deposit_thickness.tif')
        else:
            src = os.path.join(path_data, 'SoilGrids', 'BDTICM_M_1km_ll.tif')
        out = os.path.join(path_intrim, f'temp_avgDTB{i}.tif')

        # subset to CONUS region
        subprocess.run(
            ["gdal_translate", "-projwin", "-125.82", "49.5", "-66.17", "24.4",
            "-co", "COMPRESS=LZW", src, out],
            check=True
        )

        with rio.open(out) as h:
            lat_, lon_ = get_latlon(out)
            data = h.read(1, masked = True)
            data = np.where(data.mask, np.nan, data.data)

        lon2d, lat2d = np.meshgrid(lon, lat)

        f = RegularGridInterpolator((lat_ ,lon_), data, 
                                    method = 'nearest', bounds_error=False,
                                    fill_value = np.nan)
        data_interp[i] = f(np.column_stack([lat2d.ravel(), lon2d.ravel()])).reshape(lat2d.shape)
        
        if i == 1:
            data_interp[i] /= 100

    return data_interp


if __name__ == '__main__':
    tif = os.path.join(path_data, 'POLARIS', 'polaris_clay_1km.tif')
    lat, lon = get_latlon(tif)
    create_file(lat, lon)

    data = read_polaris('pH')
    append_layer(data, 'pH', -9999.0, 
                 long_name = 'Soil pH', units = '')

    data = read_polaris('PCT_CLAY')
    append_layer(data, 'PCT_CLAY', -9999.0, 
                 long_name = 'percentage clay content', units = '%')

    data = read_polaris('PCT_SAND')
    append_layer(data, 'PCT_SAND', -9999.0, 
                 long_name = 'percentage sand content', units = '%')

    data = read_polaris('bd_col')
    append_layer(data, 'bd_col', -9999.0, 
                 long_name = 'bulk density', units = 'kg/m3')
    
    cfvo = read_cfvo(lat, lon)
    append_layer(cfvo, 'cfvo', -9999.0, 
                 long_name = 'volume fraction of coarse fragments', units = '%')

    data = read_polaris('hksat_col')
    append_layer(data, 'hksat_col', -9999.0, 
                 long_name = 'saturated hydraulic conductivity', units = 'mm H2O/s')

    data = read_polaris('watsat_col')
    append_layer(data, 'watsat_col', -9999.0, 
                 long_name = 'saturation water content', units = 'm3/m3')

    data = read_polaris('ORGANIC', cfvo)
    append_layer(data, 'ORGANIC', -9999.0, 
                 long_name = 'organic matter content', units = 'kg/m3')

    data = read_organic2(lat,lon)
    append_layer(data, 'ORGANIC2', -9999.0, 
                 long_name = 'organic matter content from Li et al.', units = 'kg/m3')

    data = read_gNATSGO('ORGANIC', lat, lon, cfvo)
    append_layer(data, 'ORGANIC3', -9999.0, 
                 long_name = 'organic matter content', units = 'kg/m3')

    data_interp = read_aveDTB(lat, lon)
    append_layer(data_interp[0], 'aveDTB', -9999.0,
                 long_name = 'soil thickness from Pelletier et al.', units = 'm')
    append_layer(data_interp[1], 'aveDTB2', -9999.0,
                 long_name = 'soil thickness from SoilGrids', unit = 'm')