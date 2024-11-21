#!/bin/csh -f
#SBATCH --time=12:00:00
#SBATCH -J temp_gNATSGO_geotiff_REPLACE
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --mem=128G
#SBATCH --ntasks-per-node 1
#SBATCH --export=ALL
#SBATCH -A m2467
#SBATCH --constraint=cpu

echo `which python`
cd ~/Git/soil_properties
python temp_gNATSGO_geotiff_REPLACE.py
