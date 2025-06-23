from osgeo import gdal,ogr,osr

bb = (-337500.000,1242500.000,152500.000,527500.000) # Example bounding box (homolosine) for Ghana
igh = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs" # proj string for Homolosine projection
res = 250 
location = "https://files.isric.org/soilgrids/latest/data/"
sg_url = f"/vsicurl?max_retry=3&retry_delay=1&list_dir=no&url={location}"

kwargs = {'format': 'GTiff', 'projWin': bb, 'projWinSRS': igh, 'xRes': res, 'yRes': res}

ds = gdal.Translate('./crop_roi_igh_py.vrt', 
                    '/vsicurl?max_retry=3&retry_delay=1&list_dir=no&url=https://files.isric.org/soilgrids/latest/data/ocs/ocs_0-30cm_mean.vrt', 
                    **kwargs)
del ds

ds = gdal.Warp('./crop_roi_ll_py.vrt', 
    './crop_roi_igh_py.vrt', 
    dstSRS='EPSG:4326')
del ds

kwargs = {'format': 'GTiff', 'creationOptions': ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2", "BIGTIFF=YES"] }
ds = gdal.Translate('./crop_roi_ll_py.tif',
    './crop_roi_ll_py.vrt', 
    **kwargs)

del ds
