#!/bin/bash
gpu_id=3
train_dynamic=1
missing_rate=0.2
seed=2021

for model in PMR_4
do
bash run2.sh $gpu_id $model 128 $seed $missing_rate $train_dynamic
done