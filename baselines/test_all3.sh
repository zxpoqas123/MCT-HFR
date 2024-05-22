#!/bin/bash
gpu_id=2
seed=2023
train_dynamic=0
missing_rate=0.0
test_root=./train_dynamic$train_dynamic/missing_rate$missing_rate/seed_2021

dataset_type=IEMOCAP_4
for model in {MulT_4,MISA_4,CHFN_4}
do
    bash test.sh $gpu_id $model 128 $seed $test_root $dataset_type
done
bash test.sh $gpu_id TFN_4 64 $seed $test_root $dataset_type

dataset_type=MSP-IMPROV
for model in {MulT_4,MISA_4,CHFN_4}
do
    bash test.sh $gpu_id $model 128 $seed $test_root $dataset_type
done
bash test.sh $gpu_id TFN_4 64 $seed $test_root $dataset_type