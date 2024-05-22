#!/bin/bash
max_len='400,40,50'
seed=2021
for missing_rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
for dataset_type in IEMOCAP_4 MSP-IMPROV
do
    python generate_mask.py \
        --seed $seed \
        --dataset_type $dataset_type \
        --max_len $max_len \
        --missing_rate $missing_rate
done
done