#!/bin/bash
gpu_id=3
train_dynamic=1
missing_rate=0.5
test_root=./train_dynamic$train_dynamic/missing_rate$missing_rate/seed_$seed

dataset_type=IEMOCAP_4
for seed in {2021..2023..1}
do
for model in {MCT_4,TFRNet_4}
do
    test_root=./train_dynamic$train_dynamic/missing_rate$missing_rate/seed_$seed
    bash test.sh $gpu_id $model 128 $seed $test_root $dataset_type
done
done

dataset_type=MSP-IMPROV
for seed in {2021..2023..1}
do
for model in {MCT_4,TFRNet_4}
do
    test_root=./train_dynamic$train_dynamic/missing_rate$missing_rate/seed_$seed
    bash test.sh $gpu_id $model 128 $seed $test_root $dataset_type
done
done
