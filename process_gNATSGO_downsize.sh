#!/bin/csh -f
#SBATCH --time=2:00:00
#SBATCH -J process_gNATSGO_downsize
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --mem=64G
#SBATCH --ntasks-per-node 1
#SBATCH --export=ALL
#SBATCH -A m2467
#SBATCH --constraint=cpu

echo `which python`
cd ~/Git/soil_properties
python process_gNATSGO_downsize.py
