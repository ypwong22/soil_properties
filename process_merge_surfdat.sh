#!/bin/csh
#SBATCH --time=24:00:00
#SBATCH -J process_merge_surfdat
#SBATCH --nodes=1
#SBATCH -A CLI185
#SBATCH -p batch_ccsi
#SBATCH --ntasks-per-node 1

#setenv LD_LIBRARY_PATH /sw/baseline/spack-envs/base/opt/linux-rhel8-zen3/gcc-12.2.0/netlib-lapack-3.11.0-lpwyqsehj7wuz2i45umfhwa5ymv2dz5b/lib64:/sw/baseline/spack-envs/base/opt/linux-rhel8-zen3/gcc-12.2.0/openmpi-4.0.4-bxes2wvty3q7v55qep7hiuud6rocd4bl/lib:/sw/baseline/gcc/12.2.0/lib64:/ccsopen/home/zdr/opt/lib:${HOME}/.conda/envs/olmt/lib

# From load-balancing perspective, 128 cores seems inefficient. Fewer cores are better
cd ${HOME}/Git/soil_properties
${HOME}/.conda/envs/myCondaEnv/bin/python process_merge_surfdat.py
