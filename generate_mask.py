import os
import pandas as pd
import torch
import argparse
import numpy as np
import random

parser = argparse.ArgumentParser(description="Generate static missing masks for given dataset and feature.")
parser.add_argument('--seed',default=42,type=int)
parser.add_argument("--max_len",type=str,required=True)
parser.add_argument("--dataset_type",type=str,required=True)
parser.add_argument('--missing_rate',default=0.5,type=float)
args = parser.parse_args()

dataset_type=args.dataset_type
seed=args.seed
max_len_ls=[int(x) for x in args.max_len.split(',')]
missing_rate=args.missing_rate
df = pd.read_csv('/home/chenchengxin/TOMM/meta/{}.tsv'.format(dataset_type), sep='\t')
dataset_type = dataset_type.split('_')[0]
if not os.path.exists('/home/chenchengxin/TOMM/dataset/missing_masks'):
    os.makedirs('/home/chenchengxin/TOMM/dataset/missing_masks')
target_root = '/home/chenchengxin/TOMM/dataset/missing_masks/{}_{}_{}.pt'.format(dataset_type, seed, missing_rate)

def setup_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

missing_masks = []
if missing_rate <=1.0:
    for i in range(len(max_len_ls)):
        missing_masks.append(torch.rand(len(df),max_len_ls[i]) < missing_rate)

missing_dic = {}
count = 0
for i in range(len(df)):
    fn = df.filename[i]
    masks = [x[i,:] for x in missing_masks]
    missing_dic[fn] = masks
    count+=1
    if count%500 == 0:
        print('preprocessed: {}'.format(count))        

torch.save(missing_dic,target_root)
