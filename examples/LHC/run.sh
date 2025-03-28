#!/bin/bash

# Define the list of anomaly possibilities
anomaly_possibilities=(10000 2500 1750 1500 1400 1300 1250 1100 1000 500 0)

Kmax=1000
# Loop over the anomaly possibilities
for anomaly_size in "${anomaly_possibilities[@]}"
do
    echo "Running with anomaly size: $anomaly_size"
    python LHC_Olympics2020_EagleEye_DADM.py $anomaly_size $Kmax

done