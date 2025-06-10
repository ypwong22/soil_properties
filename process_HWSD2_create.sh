create=1
submit=1
clean=0

if [[ $create == 1 ]]; then
  for layer in {1..8}; do
    sed "s/REPLACE/${layer}/g" process_HWSD2.py > temp_HWSD2_${layer}.py
    sed "s/REPLACE/${layer}/g" process_HWSD2.sh > temp_HWSD2_${layer}.sh
  done
fi

if [[ $submit == 1 ]]; then
  for layer in {1..8}; do
    sbatch temp_HWSD2_${layer}.sh
  done
fi

if [[ $clean == 1 ]]; then
  for layer in {1..8}; do
    rm temp_HWSD2_${layer}.py
    rm temp_HWSD2_${layer}.sh
  done
fi
