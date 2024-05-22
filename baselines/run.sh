#!/bin/bash
epochs=50
lr=1e-4
loss_type=CE
batch_size=32
feature='wavlm,denseface,bert'
max_len='400,50,50'
optimizer_type=AdamW
dropout=0.25

set -ue
# training the network
gpu_id=$1
model_type=$2
hidden=$3
seed=$4
missing_rate=$5
train_dynamic=$6

dataset_type=IEMOCAP_4
for fold in {0..9..2}
do
    store_root=./train_dynamic$train_dynamic/missing_rate$missing_rate/seed_$seed/$model_type+$dataset_type+$max_len\s+$feature+lr$lr+batch_size$batch_size+$loss_type+$optimizer_type+dropout$dropout+hidden$hidden/
    echo "============training fold $fold============"
    python ./train.py \
        --seed $seed \
        --max_len $max_len \
        --lr $lr \
        --epochs $epochs \
        --fold $fold \
        --root $store_root \
        --loss_type $loss_type \
        --batch_size $batch_size \
        --dataset_type $dataset_type \
        --optimizer_type $optimizer_type \
        --device_number $gpu_id \
        --feature $feature \
        --model_type $model_type \
        --dropout $dropout \
        --hidden $hidden  \
        --missing_rate $missing_rate \
        --train_dynamic $train_dynamic
done

test_root=./train_dynamic$train_dynamic/missing_rate$missing_rate/seed_$seed
bash test.sh $gpu_id $model_type $hidden $seed $test_root $dataset_type