#!/bin/bash

# Initialize GRASS GIS environment
source ~/local/grass-gis/grass-env.sh

# Initialize GRASS GIS session
# Replace these paths with your actual GRASS GIS database location and mapset
export LOCATION="gNATSGO_mukey"
export MAPSET="PERMANENT"
export GISDBASE="${PROJDIR}/GRASS_GIS"  # Your GRASS GIS database directory

# Create location if it doesn't exist
if [ ! -d "$GISDBASE/$LOCATION" ]; then
    grass -c epsg:5070 "$GISDBASE/$LOCATION"
    grass -c epsg:4326 "$GISDBASE/${LOCATION}_reproj"
fi

# Set custom variables
export INPUTDIR=${PROJDIR}/Soil_Properties/intermediate/gNATSGO/
export NUM_BANDS=10

# ~1 hour per variable per band
export VARNAME=silttotal_r
#for VARNAME in sandtotal_r silttotal_r claytotal_r om_r ksat_r brockdepmin
#for band in $(seq 1 $NUM_BANDS); do

for i in {1..10}; do
    export band=${i}
    echo ${band}

    # Start GRASS session
    grass "$GISDBASE/$LOCATION/$MAPSET" --exec bash << 'EOF'

######################################################################
# Run the following inside the GRASS GIS interface

# Import the files to mosaic
for file in ${INPUTDIR}/mukey_${VARNAME}*.tif; do 
    base=$(basename "$file" .tif)
    r.in.gdal input="$file" output="temp_${base}_${band}" band=$band --overwrite
done

# Set the map's spatial extent
band_files=$(g.list type=raster pattern="temp*${VARNAME}*${band}" separator=,)
g.region raster=${band_files} -b

# Patch (mosaic) all imported rasters; do not need to set memory if running on a node
r.patch input=$band_files output="mosaic_${VARNAME}_${band}" --overwrite

# Check the region using lat lon units
g.region raster=mosaic_${VARNAME}_${band} -l

# Clean up temporary files
g.remove -f type=raster pattern="temp_*${VARNAME}*${band}"

# Switch to target location
g.mapset -c location=${LOCATION}_reproj mapset=${MAPSET}

# Set region to target resolution and extent
g.region n=50 s=24.5 e=-65.5 w=-127.5 res=0.008333

# Reproject and resample using bilinear interpolation
# (memory is specified in megabytes)
r.proj location=$LOCATION mapset=$MAPSET input="mosaic_${VARNAME}_${band}" \
    output="mosaic_${VARNAME}_${band}" method=bilinear --overwrite # memory=8192 

# Export to NetCDF
r.out.gdal input="mosaic_${VARNAME}_${band}" \
    output="${INPUTDIR}/${VARNAME}_${band}.nc" format=netCDF \
    createopt="FORMAT=NC4,COMPRESS=DEFLATE" --overwrite
######################################################################

EOF

    #band=$((band + 1))

done
#done
