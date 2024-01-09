#!/bin/bash
epochs=40
lr=1e-4
loss_type=CE
batch_size=32
feature='wav2vec,vggface,bert'
max_len='400,40,50'
optimizer_type=AdamW
dropout=0.25

set -ue
# training the network
gpu_id=$1
model_type=$2
hidden=$3
seed=$4
test_root=$5
dataset_type=$6
duration=$7
for missing_rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    python test.py \
        --duration $duration \
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