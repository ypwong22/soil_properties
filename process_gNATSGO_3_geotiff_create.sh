create=1
submit=1
clean=0

if [[ $create == 1 ]]; then
  for chunk in {1..72}; do
    sed "s/REPLACE/${chunk}/g" process_gNATSGO_3_geotiff.py > temp_gNATSGO_geotiff_${chunk}.py
    sed "s/REPLACE/${chunk}/g" process_gNATSGO_3_geotiff.sh > temp_gNATSGO_geotiff_${chunk}.sh
  done
fi

if [[ $submit == 1 ]]; then
  for chunk in {1..72}; do
    sbatch temp_gNATSGO_geotiff_${chunk}.sh
  done
fi

if [[ $clean == 1 ]]; then
  for chunk in {1..72}; do
    rm temp_gNATSGO_geotiff_${chunk}.py
    rm temp_gNATSGO_geotiff_${chunk}.sh
  done
fi
