#!/bin/bash
gpu_id=3
seed=2021
train_dynamic=1
missing_rate=0.2
test_root=./train_dynamic$train_dynamic/missing_rate$missing_rate/seed_2021

dataset_type=IEMOCAP_4
bash test_m.sh $gpu_id MISA_4 128 $seed $test_root $dataset_type

dataset_type=MSP-IMPROV
bash test_m.sh $gpu_id MISA_4 128 $seed $test_root $dataset_type