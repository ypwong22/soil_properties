""" Convert the gNATSGO to 1km NetCDF file
    Raw GPT-generated script. Not yet tried!
"""
from osgeo import gdal, osr
import numpy as np
import xarray as xr
import os
from tqdm import tqdm
import tempfile
import scipy.interpolate


def get_inputs(var):
    # 1,73
    return [os.path.join(path_root, f'mukey_{var}_{i}.tif') for i in range(1,3)]

def get_latlon():
    conus_bounds = [23.0,54.5,-125.5,-66.5]
    target_lats = np.linspace(-89.995834, 89.995834, 21600)
    target_lons = np.linspace(-179.99583, 179.99583, 43200)
    target_lats = target_lats[(target_lats <= conus_bounds[1]) & (target_lats >= conus_bounds[0])]
    target_lons = target_lons[(target_lons <= conus_bounds[3]) & (target_lons >= conus_bounds[2])]
    return target_lats, target_lons


path_root = os.path.join(os.environ['PROJDIR'], 'DATA', 'Soil_Properties', 
                         'gNATSGO_CONUS', 'mukey_divide')
input_list = get_inputs('sandtotal_r')
target_lats, target_lons = get_latlon()


#############################################################################
# Combine the individual geotiff files into VRT
#############################################################################
vrt_path = tempfile.mktemp(suffix = '.vrt', prefix = path_root)
vrt_options = gdal.BuildVRTOptions(resampleAlg = 'bilinear')
gdal.BuildVRT(vrt_path, input_list, options = vrt_options)


#############################################################################
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


from osgeo import gdal, osr
import numpy as np
import xarray as xr
import os
from tqdm import tqdm
import tempfile

class GDALProcessor:
    def __init__(self, input_files, target_lats, target_lons):
        self.input_files = input_files
        self.target_lats = target_lats
        self.target_lons = target_lons
        
        # Configure GDAL
        gdal.UseExceptions()
        gdal.SetCacheMax(1024 * 1024 * 512)  # Set cache to 512MB

    def create_warped_vrt(self):
        """Create a warped VRT that handles both mosaicking and reprojection."""
        # Create initial VRT from input files
        temp_vrt = tempfile.mktemp(suffix='.vrt')
        vrt_options = gdal.BuildVRTOptions(resampleAlg='bilinear')
        gdal.BuildVRT(temp_vrt, self.input_files, options=vrt_options)
        
        # Set up coordinate transformation
        source = osr.SpatialReference()
        source.ImportFromEPSG(5070)
        
        target = osr.SpatialReference()
        target.ImportFromEPSG(4326)
        
        # Calculate output dimensions based on target coordinates
        pixel_size_y = (max(self.target_lats) - min(self.target_lats)) / len(self.target_lats)
        pixel_size_x = (max(self.target_lons) - min(self.target_lons)) / len(self.target_lons)
        
        # Create warped VRT
        warped_vrt = tempfile.mktemp(suffix='.vrt')
        gdal.Warp(warped_vrt, temp_vrt,
                 dstSRS='EPSG:4326',
                 xRes=pixel_size_x,
                 yRes=pixel_size_y,
                 outputBounds=[min(self.target_lons), min(self.target_lats),
                              max(self.target_lons), max(self.target_lats)],
                 resampleAlg=gdal.GRA_Bilinear)
        
        # Clean up temporary VRT
        os.remove(temp_vrt)
        return warped_vrt

    def process_dataset(self, output_path):
        """Process the entire dataset with proper interpolation."""
        print("Creating warped virtual dataset...")
        warped_vrt_path = self.create_warped_vrt()
        
        try:
            # Open warped VRT
            ds = gdal.Open(warped_vrt_path)
            
            # Initialize output array
            band_count = ds.RasterCount
            output_data = np.zeros((band_count,
                                  len(self.target_lats),
                                  len(self.target_lons)))
            
            print("Reading and interpolating data...")
            # Read and process each band
            for band_idx in tqdm(range(band_count)):
                band = ds.GetRasterBand(band_idx + 1)
                
                # Read the entire band
                data = band.ReadAsArray()
                nodata = band.GetNoDataValue()
                
                # Mask nodata values
                if nodata is not None:
                    data = np.ma.masked_equal(data, nodata)
                
                # The data is already interpolated to the target grid by GDAL
                output_data[band_idx] = data
            
            print("Saving to NetCDF...")
            # Create and save NetCDF
            ds_out = xr.Dataset(
                {
                    'data': (['band', 'lat', 'lon'], output_data),
                },
                coords={
                    'lat': self.target_lats[::-1],  # Reverse lats to match GDAL's output
                    'lon': self.target_lons,
                    'band': np.arange(band_count)
                }
            )
            ds_out.to_netcdf(output_path)
            
        finally:
            # Clean up
            ds = None
            if os.path.exists(warped_vrt_path):
                os.remove(warped_vrt_path)

def main():
    input_files = [f for f in os.listdir('.') if f.endswith('.tif')]
    target_lats = np.linspace(start_lat, end_lat, num_lat_points)
    target_lons = np.linspace(start_lon, end_lon, num_lon_points)
    
    processor = GDALProcessor(input_files, target_lats, target_lons)
    processor.process_dataset('output.nc')

if __name__ == "__main__":
    main()