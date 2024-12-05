BOUNDS="-337500.000 1242500.000 152500.000 527500.000" # Example bounding box (homolosine) for Ghana
ulx uly lrx lry
CELL_SIZE="250 250"

IGH="+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs" # proj string for Homolosine projection
SG_URL="/vsicurl?max_retry=3&retry_delay=1&list_dir=no&url=https://files.isric.org/soilgrids/latest/data"


gdal_translate -of VRT -projwin $BOUNDS -tr $CELL_SIZE \
    -co "TILED=YES" -co "COMPRESS=DEFLATE" -co "PREDICTOR=2" -co "BIGTIFF=YES" \
    "/vsicurl?max_retry=3&retry_delay=1&list_dir=no&url=https://files.isric.org/soilgrids/latest/data/ocs_0-30cm_mean.vrt" \
    "ocs_0-5cm_mean.vrt"

gdalwarp -overwrite -t_srs EPSG:4326 -of VRT "ocs_0-5cm_mean.vrt" "ocs_0-5cm_mean_4326.vrt"


gdal_translate ocs_0-5cm_mean_4326.vrt ocs_0-5cm_mean_4326.tif \
    -co "TILED=YES" -co "COMPRESS=DEFLATE" -co "PREDICTOR=2" -co "BIGTIFF=YES"
