#!/bin/bash
seed=2021
max_len='400,50,50'
for missing_rate in {2.0,3.0,4.0,5.0,6.0,7.0}
do
for dataset_type in {IEMOCAP_4,MSP-IMPROV}
do
    python generate_mask.py \
        --seed $seed \
        --dataset_type $dataset_type \
        --max_len $max_len \
        --missing_rate $missing_rate
done
done