## MCT-HFR

This repository contains the implementation of the paper "Modality-collaborative Transformer with Hybrid Feature Reconstruction for Robust Emotion Recognition". 

### Missing mask preparation (for one-to-one training):

bash prepare_missing_mask.sh

### Configuration statement:

gpu_id=0 (indicating which gpu to use) 

model=MCT_4 (indicating which model type to use. The number 4 indicates 4 stacked layers, which can be replaced by any other number)

hidden=128 (indicating the hidden dimensions of Transformer layer)

seed=2021 (indicating the random seed)

missing_rate=0.2 (indicating the missing rate of input features)

train_dynamic=1 (indicating the dynamic training. Switching it to 0 will indicate the static training, i.e., one-to-one training)

### Model training:

bash run.sh $gpu_id $model $hidden $seed $missing_rate $train_dynamic 

### Model evaluation:

bash test.sh $gpu_id $model $hidden $seed $missing_rate $train_dynamic 


