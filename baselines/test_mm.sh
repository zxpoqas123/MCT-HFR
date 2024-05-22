#!/bin/bash
lr=1e-4
loss_type=CE
batch_size=32
feature='wavlm,denseface,bert'
max_len='400,50,50'
optimizer_type=AdamW
dropout=0.25

set -ue
# training the network
gpu_id=0
model_type=PMR_4
hidden=128
seed=2021
dataset_type=IEMOCAP_4
train_dynamic=0

for missing_rate in {2.0,3.0,4.0,5.0,6.0,7.0}
do
    test_root=./train_dynamic$train_dynamic/missing_rate$missing_rate/seed_2021
    python test_m.py \
        --seed $seed \
        --model_type $model_type \
        --dataset_type $dataset_type \
        --device_number $gpu_id \
        --batch_size $batch_size \
        --feature $feature \
        --max_len $max_len \
        --optimizer_type $optimizer_type \
        --lr $lr \
        --dropout $dropout \
        --hidden $hidden   \
        --missing_rate $missing_rate \
        --test_root $test_root 
done