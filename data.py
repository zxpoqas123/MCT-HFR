import os
import numpy as np
import pandas as pd
from pandas import Series
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Pad_trunc_seq(torch.nn.Module):
    """Pad or truncate a sequence data to a fixed length.
    Args:
      x: ndarray, input sequence data.
      max_len: integer, length of sequence to be padded or truncated.
    """
    def __init__(self, max_len: int = 50, keep_ori=False):
        super(Pad_trunc_seq, self).__init__()
        self.max_len = max_len
        self.keep_ori = keep_ori
    def forward(self,x):
        shape = x.shape
        length = shape[0]
        if not self.keep_ori:
            mask_padding = torch.ones(self.max_len)
            mask_padding[:length] = 0
            mask_padding = torch.logical_or(mask_padding, torch.zeros(self.max_len))
            if length < self.max_len:    
                pad_shape = (self.max_len - length,) + shape[1:]
                pad = torch.zeros(pad_shape)
                x_new = torch.cat((x, pad), axis=0)
            else:
                x_new = x[0:self.max_len,:]
        else:
            x_new = x
            mask_padding = torch.logical_not(torch.ones(length))
        return x_new, mask_padding

class Augment(object):
    def __init__(self, max_len=50):
        super(Augment, self).__init__()
        self.feature = feature
        self.pad_trunc_seq = Pad_trunc_seq(max_len = max_len)

    def __call__(self, x):
        return self.pad_trunc_seq(x)

class Mydataset(Dataset):
    def __init__(self, dataset_type='IEMOCAP_4', mode='train', max_len_ls=[400,40,50], fold=0, feature_ls=['wav2vec','vggface','bert'], missing_rate=0.5, seed=2021, train_dynamic=0, duration='all'):
        data_all = pd.read_csv('/home/chenchengxin/TOMM/meta/{}.tsv'.format(dataset_type), sep='\t')
        SpkNames = np.unique(data_all['speaker'])
        self.mode = mode
        self.duration = duration
        self.train_dynamic = train_dynamic
        self.missing_rate = missing_rate
        self.seed = seed
        self.data_info = self.split_dataset(data_all, fold, SpkNames)
        dataset_type = dataset_type.split('_')[0]
        feature_detail_dic={'wav2vec':'Wav2Vec2_base_512','vggface':'VggFace2_512','bert':'BERT_base_768'}

        self.modalities = len(feature_ls)
        self.feature_root = ['/home/chenchengxin/TOMM/dataset/{}_{}.pt'.format(dataset_type, feature_detail_dic[feature]) for feature in feature_ls]
        self.mask_missing_root = '/home/chenchengxin/TOMM/dataset/missing_masks/{}_{}_{}.pt'.format(dataset_type, seed, missing_rate)
        self.mask_missing_dic = torch.load(self.mask_missing_root)
        self.feature_dic_ls = []
        self.transform_ls = []
        for i in range(self.modalities):
            self.feature_dic_ls.append(torch.load(self.feature_root[i]))       
            self.transform_ls.append(transforms.Compose([Augment(max_len=max_len_ls[i], feature=feature_ls[i])]))

        self.label = self.data_info['label'].astype('category').cat.codes.values
        self.speaker = self.data_info['speaker'].astype('category').cat.codes.values
        self.ClassNames = np.unique(self.data_info['label'])
        self.SpeakerNames = np.sort(np.unique(self.data_info['speaker']))
        self.NumClasses = len(self.ClassNames)
        self.NumSpeakers = len(self.SpeakerNames)
        self.weight = 1/torch.tensor([(self.label==i).sum() for i in range(self.NumClasses)]).float()

    def split_dataset(self, df_all, fold, speakers):
        spk_len = len(speakers)
        val_idx = np.array(df_all['speaker']==speakers[(fold*2)%spk_len])
        test_idx = np.array(df_all['speaker']==speakers[(fold*2+1)%spk_len])
        train_idx = True^(test_idx+val_idx)
        if self.duration == 'g1':
            dur_idx = np.array(df_all['duration']<=2.0)
        elif self.duration == 'g2':
            dur_idx = np.logical_and(np.array(df_all['duration']>2.0), np.array(df_all['duration']<=4.0))
        elif self.duration == 'g3':
            dur_idx = np.logical_and(np.array(df_all['duration']>4.0), np.array(df_all['duration']<=6.0))       
        elif self.duration == 'g4':
            dur_idx = np.logical_and(np.array(df_all['duration']>6.0), np.array(df_all['duration']<=8.0))   
        elif self.duration == 'g5':
            dur_idx = np.array(df_all['duration']>8.0)  
        if not self.duration=='all':
            val_idx = np.logical_and(val_idx, dur_idx)
            test_idx = np.logical_and(test_idx, dur_idx)
        train_data_info = df_all[train_idx].reset_index(drop=True)
        val_data_info = df_all[val_idx].reset_index(drop=True)
        test_data_info = df_all[test_idx].reset_index(drop=True)

        if self.mode == 'train':
            data_info = train_data_info
        elif self.mode == 'val':
            data_info = val_data_info
        elif self.mode == 'test':
            data_info = test_data_info
        else:
            data_info = df_all
        return data_info  

    def generate_m(self, fn, mask_paddings):
        # paddings are arranged as (A,V,L): shape: [(max_len_1,),(max_len_2,),(max_len_3)]
        if self.mode=='train' and self.train_dynamic:
            mask_missings = []
            for i in range(self.modalities):
                mask_missing = torch.rand(mask_paddings[i].shape) < self.missing_rate   # Here True indicates the missing positions!             
                mask_missing = torch.logical_and(mask_missing, torch.logical_not(mask_paddings[i]))
                mask_missings.append(mask_missing)
        else:
            mask_missings = self.mask_missing_dic[fn]
            for i in range(self.modalities):
                mask_missings[i] = torch.logical_and(mask_missings[i], torch.logical_not(mask_paddings[i]))   
        return mask_missings

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        fn = self.data_info.filename[idx]
        audio, audio_mask_padding = self.transform_ls[0](self.feature_dic_ls[0][fn])
        vision, vision_mask_padding = self.transform_ls[1](self.feature_dic_ls[1][fn])
        text, text_mask_padding = self.transform_ls[2](self.feature_dic_ls[2][fn])
        audio_mask_missing, vision_mask_missing, text_mask_missing = self.generate_m(fn=fn, mask_paddings=[audio_mask_padding, vision_mask_padding, text_mask_padding])
        label = self.label[idx]
        label = np.array(label)
        label = label.astype('float').reshape(1)
        label = torch.Tensor(label).long().squeeze()
        sample = {
            'audio': audio.float(),
            'audio_mask_padding': audio_mask_padding,
            'audio_m': audio.float()*(torch.logical_not(audio_mask_missing).unsqueeze(-1)),
            'audio_mask_missing': audio_mask_missing,
            'vision': vision.float(),
            'vision_mask_padding': vision_mask_padding,
            'vision_m': vision.float()*(torch.logical_not(vision_mask_missing).unsqueeze(-1)),
            'vision_mask_missing': vision_mask_missing,
            'text': text.float(),
            'text_mask_padding': text_mask_padding,
            'text_m': text.float()*(torch.logical_not(text_mask_missing).unsqueeze(-1)),
            'text_mask_missing': text_mask_missing,
            'label': label,
            'index': idx
        }
        return sample