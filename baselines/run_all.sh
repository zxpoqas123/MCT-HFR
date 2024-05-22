#!/bin/bash
gpu_id=2
train_dynamic=0
#missing_rate=2.0
seed=2021

#bash run.sh $gpu_id MCT_4 128 $seed $missing_rate $train_dynamic
for missing_rate in {2.0,3.0,4.0}
do
bash run.sh $gpu_id PMR_4 128 $seed $missing_rate $train_dynamic
done