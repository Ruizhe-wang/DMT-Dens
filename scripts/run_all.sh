#!/bin/bash

K_list=(1000 200 100 30 20 10)
nu_lat_list=(0.001 0.002 0.01 0.02 0.1)

for K in "${K_list[@]}"; do
  for nu_lat in "${nu_lat_list[@]}"; do
    echo "Running: K=$K, nu_lat=$nu_lat"
    python main.py fit -c conf/difftree/G_xinli_1gpu.yaml --model.nu_lat "$nu_lat" --data.K "$K"
  done
done