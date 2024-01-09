## MCT-HFR

This repository contains the implementation of the paper "Modality-Collaborative Transformer with Hybrid Feature Reconstruction for Robust Emotion Recognition". 

### Missing mask preparation (for one-to-one training):

bash prepare_missing_mask.sh

### Configuration statement:

gpu_id=0 (indicates which gpu to use) 

model=MCT_4 (indicates which model type to use)

hidden=128 (indicates the hidden dimensions of Transformer layer)

seed=2021 (indicates the random seed)

missing_rate=0.2 (indicates the missing rate for training)

train_dynamic=1 (1 indicates the dynamic training, while 0 indicates the static training, i.e., one-to-one training)

### Model training:

bash run.sh $gpu_id $model $hidden $seed $missing_rate $train_dynamic 

### Model evaluation:

bash test.sh $gpu_id $model $hidden $seed $missing_rate $train_dynamic 

#### Note: We first release the codes for data preparation and model training. The pre-processed feature files are coming soon.


