#!/bin/bash -e

#SBATCH -t 1:00:00
#SBATCH -J process_HWSD2_REPLACE
#SBATCH --nodes=1
#SBATCH -A CLI185
#SBATCH -p batch_ccsi

echo `which python`
cd ~/Git/soil_properties
python temp_HWSD2_REPLACE.py
