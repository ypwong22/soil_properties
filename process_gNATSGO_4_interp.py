""" Convert the gNATSGO to 1km NetCDF file
    Raw GPT-generated script. Not yet tried!
"""
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
import numpy as np
import xarray as xr
from pyproj import Transformer
import dask.array as da
from dask.distributed import Client, LocalCluster
import os
from itertools import product
from tqdm import tqdm

class GeoTiffProcessor:
    def __init__(self, input_files, target_lats, target_lons, tile_size=1000):
        self.input_files = input_files
        self.target_lats = target_lats
        self.target_lons = target_lons
        self.tile_size = tile_size
        self.transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
        
    def get_bounds(self):
        """Calculate the total bounds of all input files."""
        bounds_list = []
        for filepath in self.input_files:
            with rasterio.open(filepath) as src:
                bounds_list.append(src.bounds)
                
        left = min(bound.left for bound in bounds_list)
        bottom = min(bound.bottom for bound in bounds_list)
        right = max(bound.right for bound in bounds_list)
        top = max(bound.top for bound in bounds_list)
        
        return rasterio.coords.BoundingBox(left, bottom, right, top)
    
    def calculate_tiles(self, total_bounds, tile_size):
        """Calculate tile coordinates for processing."""
        width = int(np.ceil((total_bounds.right - total_bounds.left) / self.profile['transform'][0]))
        height = int(np.ceil((total_bounds.top - total_bounds.bottom) / -self.profile['transform'][4]))
        
        tiles = []
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                window = Window(x, y,
                              min(tile_size, width - x),
                              min(tile_size, height - y))
                tiles.append(window)
        return tiles

    def process_tile(self, tile, temp_merged_path):
        """Process a single tile of the merged dataset."""
        with rasterio.open(temp_merged_path) as src:
            data = src.read(window=tile)
            
            # Calculate tile bounds
            window_transform = rasterio.windows.transform(tile, src.transform)
            cols, rows = tile.width, tile.height
            xs = np.linspace(window_transform[2],
                           window_transform[2] + cols * window_transform[0],
                           cols)
            ys = np.linspace(window_transform[5],
                           window_transform[5] + rows * window_transform[4],
                           rows)
            xx, yy = np.meshgrid(xs, ys)
            
            # Transform coordinates to lat/lon
            lon, lat = self.transformer.transform(xx, yy)
            
            # Perform bilinear interpolation for each band
            interpolated = []
            for band_idx in range(data.shape[0]):
                valid_mask = ~np.isnan(data[band_idx])
                if np.any(valid_mask):
                    interp_band = scipy.interpolate.griddata(
                        (lon[valid_mask].flatten(), lat[valid_mask].flatten()),
                        data[band_idx][valid_mask].flatten(),
                        (self.target_lons, self.target_lats),
                        method='linear'
                    )
                    interpolated.append(interp_band)
                else:
                    interpolated.append(np.full((len(self.target_lats), 
                                              len(self.target_lons)), np.nan))
            
            return np.stack(interpolated)

    def merge_and_process(self, output_path):
        """Main processing function that merges and interpolates the data."""
        # First, get the profile from the first file
        with rasterio.open(self.input_files[0]) as src:
            self.profile = src.profile.copy()
        
        # Calculate total bounds
        total_bounds = self.get_bounds()
        
        # Create a temporary merged file on disk
        temp_merged_path = 'temp_merged.tif'
        
        # Merge files using rasterio's merge
        print("Merging files...")
        with rasterio.Env():
            rasterio.merge.merge(self.input_files, 
                               bounds=total_bounds,
                               res=(self.profile['transform'][0], -self.profile['transform'][4]),
                               dst_path=temp_merged_path)
        
        # Calculate tiles
        tiles = self.calculate_tiles(total_bounds, self.tile_size)
        
        # Process tiles and accumulate results
        print("Processing tiles...")
        accumulated_data = np.zeros((self.profile['count'],
                                   len(self.target_lats),
                                   len(self.target_lons)))
        valid_count = np.zeros_like(accumulated_data)
        
        for tile in tqdm(tiles):
            tile_data = self.process_tile(tile, temp_merged_path)
            valid_mask = ~np.isnan(tile_data)
            accumulated_data[valid_mask] += tile_data[valid_mask]
            valid_count[valid_mask] += 1
        
        # Average the accumulated data
        final_data = np.where(valid_count > 0,
                            accumulated_data / valid_count,
                            np.nan)
        
        # Create and save NetCDF
        print("Saving to NetCDF...")
        ds = xr.Dataset(
            {
                'data': (['band', 'lat', 'lon'], final_data),
            },
            coords={
                'lat': self.target_lats,
                'lon': self.target_lons,
                'band': np.arange(final_data.shape[0])
            }
        )
        ds.to_netcdf(output_path)
        
        # Clean up
        if os.path.exists(temp_merged_path):
            os.remove(temp_merged_path)

def main():
    # Example usage
    input_files = [f for f in os.listdir('.') if f.endswith('.tif')]
    target_lats = np.linspace(start_lat, end_lat, num_lat_points)
    target_lons = np.linspace(start_lon, end_lon, num_lon_points)
    
    processor = GeoTiffProcessor(input_files, target_lats, target_lons)
    processor.merge_and_process('output.nc')

if __name__ == "__main__":
    main()