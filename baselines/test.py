import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import datetime
import sys, subprocess
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam
#from lion_pytorch import Lion
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
#import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import logging
import pickle
from data import Mydataset
from model import FocalLoss, TFN, MulT, TFRNet, MCT, MISA, CHFN, PMR
from utils import mixup_data, mixup_criterion
import argparse
from sklearn.metrics import confusion_matrix,classification_report,recall_score,f1_score

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--duration",default=2021,type=str)
parser.add_argument("--seed",default=2021,type=int)
parser.add_argument("--device_number",default='0',type=str)
parser.add_argument('--max_len',default=5,type=str)
parser.add_argument("--feature",type=str,required=True)
parser.add_argument('--lr',default=1e-3,type=str)
parser.add_argument('--batch_size',default=32,type=int)
parser.add_argument('--hidden',default=128,type=int)
parser.add_argument('--dropout',default=0.25,type=float)
parser.add_argument("--model_type",type=str,required=True)
parser.add_argument("--dataset_type",type=str,required=True)
parser.add_argument("--loss_type",default='CE',type=str)
parser.add_argument("--optimizer_type",default='AdamW',type=str)
parser.add_argument('--add_dev', action='store_true')
parser.add_argument('--missing_rate',default=0.5,type=float)
parser.add_argument("--test_root",default='',type=str)
args = parser.parse_args()
seed = args.seed
feature_size_dic = {'fbank':80,'bert':1024,'wavlm':768,'denseface':342}
model_dic = {'TFN':TFN, 'MulT':MulT, 'TFRNet':TFRNet, 'MCT':MCT, 'MISA':MISA, 'CHFN':CHFN, 'PMR':PMR}

def setup_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def get_metrics(true_all,pred_all,target_names):
    con_mat = confusion_matrix(true_all,pred_all)
    cls_rpt = classification_report(true_all,pred_all,target_names=target_names,digits=3)

    F1_ls = f1_score(true_all,pred_all,average=None)
    UF1 = f1_score(true_all,pred_all,average='macro')
    WF1 = f1_score(true_all,pred_all,average='weighted')

    AR_ls = recall_score(true_all,pred_all,average=None)
    UAR = recall_score(true_all,pred_all,average='macro')
    WAR = recall_score(true_all,pred_all,average='weighted')
    return con_mat,cls_rpt,F1_ls,UF1,WF1,AR_ls,UAR,WAR

def test(model, device, val_loader, criterion, logger, target_names):
    model = model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    
    pred_all = np.array([],dtype=np.long)
    true_all = np.array([],dtype=np.long)
    embedding_ls = []
    with torch.no_grad():
        for data in tqdm(val_loader): 
            audio, audio_mask_padding = data['audio_m'].to(device), data['audio_mask_padding'].to(device)
            vision, vision_mask_padding = data['vision_m'].to(device), data['vision_mask_padding'].to(device)
            text, text_mask_padding = data['text_m'].to(device), data['text_mask_padding'].to(device)
            inputs, inputs_padding_mask = (audio,vision,text),(audio_mask_padding,vision_mask_padding,text_mask_padding)
            label = data['label'].to(device)

            output = model(inputs, inputs_padding_mask)
            test_loss += criterion(output['label'], label).item()
            pred = output['label'].argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            pred = output['label'].data.max(1)[1].cpu().numpy()
            true = label.data.cpu().numpy()
            pred_all = np.append(pred_all,pred)
            true_all = np.append(true_all,true)
            embedding_ls.append(output['embed'].cpu().numpy())

    test_loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)

    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), acc))

    con_mat,cls_rpt,F1_ls,UF1,WF1,AR_ls,UAR,WAR = get_metrics(true_all,pred_all,target_names)

    logger.info('Confusion Matrix:\n{}\n'.format(con_mat))
    logger.info('Classification Report:\n{}\n'.format(cls_rpt))
    #if not model_type.startswith('TFN'):
    embedding_all = np.concatenate(embedding_ls,0)
    res = {
        'AR_ls':AR_ls,
        'UAR':UAR,
        'WAR':WAR,
        'F1_ls':F1_ls,
        'UF1':UF1,
        'WF1':WF1,
        'embedding_all':embedding_all,
        'true_all':true_all,
        'pred_all':pred_all
    }
    return res

def test_epoch(model_root, fold, logger, best_model):   
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = args.batch_size
    feature_ls = args.feature.split(',')
    max_len_ls = [int(x) for x in args.max_len.split(',')]

    dev_set = Mydataset(dataset_type=args.dataset_type, mode='val', max_len_ls=max_len_ls, fold=fold, feature_ls=feature_ls, missing_rate=args.missing_rate, seed=seed, duration=args.duration)
    test_set = Mydataset(dataset_type=args.dataset_type, mode='test', max_len_ls=max_len_ls, fold=fold, feature_ls=feature_ls, missing_rate=args.missing_rate, seed=seed, duration=args.duration)  

    if '_' in args.model_type:
        model_name, model_layer = args.model_type.split('_')
        model_layer = int(model_layer)
    else:
        model_name = args.model_type
        model_layer = 4
          
    model = model_dic[model_name](feature_size=[feature_size_dic[feature] for feature in feature_ls], emotion_cls=test_set.NumClasses, h_dims=args.hidden, dropout=args.dropout, layers=model_layer)
    
    ckpt = torch.load(model_root)
    model.load_state_dict(ckpt)
    target_names = test_set.ClassNames

    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    criterion_test = nn.NLLLoss(reduction='sum')
    logger.info('testing {}'.format(best_model))
    #AR_ls,UAR,WAR,F1_ls,UF1,WF1,embedding_all,true_all,pred_all
    res_dev = test(model, device, dev_loader, criterion_test, logger, target_names)
    res_test = test(model, device, test_loader, criterion_test, logger, target_names)
    return res_dev, res_test, target_names

def main(fold_list, fold_root, metric_dic):
    subprocess.check_call(["cp", "test.py", fold_root])
    #logpath = os.path.join(fold_root, "{}_test.log".format(condition))
    logpath = os.path.join(fold_root, "test.log")
    logger = get_logger(logpath)
    if args.add_dev:
        dev_AR_ls_ls = []
        dev_UAR_ls = []
        dev_WAR_ls = []
        dev_F1_ls_ls = []
        dev_UF1_ls = []
        dev_WF1_ls = []   
        dev_pred_all = np.array([],dtype=np.long)
        dev_true_all = np.array([],dtype=np.long)

    test_AR_ls_ls = []
    test_UAR_ls = []
    test_WAR_ls = []
    test_F1_ls_ls = []
    test_UF1_ls = []
    test_WF1_ls = []
    test_pred_all = np.array([],dtype=np.long)
    test_true_all = np.array([],dtype=np.long)

    for fold in fold_list:
        logger.info('fold: {}'.format(fold))
        root = os.path.join(fold_root, 'fold_{}'.format(fold))
        model_ls = list(i for i in os.listdir(root) if i.startswith('best_epoch'))
        epoch_ls = [int(i[:-3].split('_')[-1]) for i in model_ls]
        best_model = model_ls[np.argmax(epoch_ls)]
        model_root = os.path.join(root,best_model)
        res_dev, res_test, target_names = test_epoch(model_root, fold, logger, best_model)
        
        if args.add_dev:
            if args.model_type.startswith('MCT'):
                np.save(os.path.join(root,'dev_pred_missing_rate{}.npy'.format(args.missing_rate)),res_dev['pred_all'])
                np.save(os.path.join(root,'dev_embedding_missing_rate{}.npy'.format(args.missing_rate)),res_dev['embedding_all'])
                np.save(os.path.join(root,'dev_label.npy'),res_dev['true_all'])
            dev_AR_ls_ls.append(res_dev['AR_ls'])
            dev_UAR_ls.append(res_dev['UAR'])
            dev_WAR_ls.append(res_dev['WAR'])        
            dev_F1_ls_ls.append(res_dev['F1_ls'])
            dev_UF1_ls.append(res_dev['UF1'])
            dev_WF1_ls.append(res_dev['WF1'])        
            dev_pred_all = np.append(dev_pred_all,res_dev['pred_all'])
            dev_true_all = np.append(dev_true_all,res_dev['true_all'])

        if args.model_type.startswith('MCT'):
            np.save(os.path.join(root,'test_pred_missing_rate{}.npy'.format(args.missing_rate)),res_test['pred_all'])
            np.save(os.path.join(root,'test_embedding_missing_rate{}.npy'.format(args.missing_rate)),res_test['embedding_all'])
            np.save(os.path.join(root,'test_label.npy'),res_test['true_all'])
        test_AR_ls_ls.append(res_test['AR_ls'])
        test_UAR_ls.append(res_test['UAR'])
        test_WAR_ls.append(res_test['WAR'])        
        test_F1_ls_ls.append(res_test['F1_ls'])
        test_UF1_ls.append(res_test['UF1'])
        test_WF1_ls.append(res_test['WF1'])
        test_pred_all = np.append(test_pred_all,res_test['pred_all'])
        test_true_all = np.append(test_true_all,res_test['true_all'])

    AR_ls_ls = dev_AR_ls_ls + test_AR_ls_ls if args.add_dev else test_AR_ls_ls
    UAR_ls = dev_UAR_ls + test_UAR_ls if args.add_dev else test_UAR_ls
    WAR_ls = dev_WAR_ls + test_WAR_ls if args.add_dev else test_WAR_ls
    F1_ls_ls = dev_F1_ls_ls + test_F1_ls_ls if args.add_dev else test_F1_ls_ls
    UF1_ls = dev_UF1_ls + test_UF1_ls if args.add_dev else test_UF1_ls
    WF1_ls = dev_WF1_ls + test_WF1_ls if args.add_dev else test_WF1_ls

    pred_all = np.concatenate([dev_pred_all,test_pred_all]) if args.add_dev else test_pred_all
    true_all = np.concatenate([dev_true_all,test_true_all]) if args.add_dev else test_true_all
    con_mat,cls_rpt,F1_ls,UF1,WF1,AR_ls,UAR,WAR = get_metrics(true_all,pred_all,target_names)

    logger.info('Hyper-parameters: {}\n'.format(args))
    logger.info('AR_ls_test list: {}'.format(AR_ls_ls))
    logger.info('UAR_test list: {}'.format(UAR_ls))
    logger.info('WAR_test list: {}'.format(WAR_ls))
    logger.info('AR_ls_test avg: {}'.format(np.mean(AR_ls_ls,0)))
    logger.info('UAR_test avg: {:.4f}'.format(np.mean(UAR_ls)))
    logger.info('WAR_test avg: {:.4f}'.format(np.mean(WAR_ls)))
    logger.info('F1_ls_test list: {}'.format(F1_ls_ls))
    logger.info('UF1_test list: {}'.format(UF1_ls))
    logger.info('WF1_test list: {}'.format(WF1_ls))
    logger.info('F1_ls_test avg: {}'.format(np.mean(F1_ls_ls,0)))
    logger.info('UF1_test avg: {:.4f}'.format(np.mean(UF1_ls)))
    logger.info('WF1_test avg: {:.4f}'.format(np.mean(WF1_ls)))

    logger.info('global-wise metrics:')
    logger.info('Confusion Matrix:\n{}\n'.format(con_mat))
    logger.info('Classification Report:\n{}\n'.format(cls_rpt))
    logger.info('AR_ls_test avg: {}'.format(AR_ls))
    logger.info('UAR_test avg: {}'.format(UAR))
    logger.info('WAR_test avg: {}'.format(WAR))
    logger.info('F1_ls_test avg: {}'.format(F1_ls))
    logger.info('UF1_test avg: {}'.format(UF1))
    logger.info('WF1_test avg: {}'.format(WF1))

    metric_dic['UAR'][args.missing_rate] = UAR
    metric_dic['WAR'][args.missing_rate] = WAR
    metric_dic['UF1'][args.missing_rate] = UF1
    metric_dic['WF1'][args.missing_rate] = WF1

    with open(os.path.join(args.test_root, 'test_all_seed_{}_{}_{}.txt'.format(seed,args.feature,args.model_type)),'a') as f:
        f.write('Hyper-parameters: {}\n'.format(args))
        f.write('global-wise metrics:\n')
        f.write('Confusion Matrix:\n{}\n'.format(con_mat))
        f.write('Classification Report:\n{}\n'.format(cls_rpt))
        f.write('AR_ls_test avg: {}\n'.format(AR_ls))
        f.write('UAR_test avg: {:.4f}\n'.format(UAR))
        f.write('WAR_test avg: {:.4f}\n'.format(WAR)) 
        f.write('F1_ls_test avg: {}\n'.format(F1_ls))
        f.write('UF1_test avg: {:.4f}\n'.format(UF1))
        f.write('WF1_test avg: {:.4f}\n'.format(WF1))        
        f.write('='*40+'\n')
        if args.missing_rate>0.85:
            f.write('UAR_dic: {}\n'.format(metric_dic['UAR']))
            f.write('WAR_dic: {}\n'.format(metric_dic['WAR']))
            f.write('UF1_dic: {}\n'.format(metric_dic['UF1']))
            f.write('WF1_dic: {}\n'.format(metric_dic['WF1']))

            f.write('UAR_AUILC: {:.4f}\n'.format(cal_AUILC(metric_dic['UAR'])))
            f.write('WAR_AUILC: {:.4f}\n'.format(cal_AUILC(metric_dic['WAR'])))
            f.write('UF1_AUILC: {:.4f}\n'.format(cal_AUILC(metric_dic['UF1'])))
            f.write('WF1_AUILC: {:.4f}\n'.format(cal_AUILC(metric_dic['WF1'])))
    return metric_dic            
    #if int(fold)==11:
    #    sys.stdout.flush()
    #    input('[Press Any Key to start another run]')

def cal_AUILC(dic):
    values = np.array(list(dic.values()),dtype=np.float)
    AUILC = (values[:-1].sum()+values[1:].sum())*0.1/2
    return AUILC

if __name__ == '__main__':
    #condition = 'impro'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    if args.dataset_type.startswith('IEMOCAP'):
        fold_list = list(range(0,10,2))
    elif args.dataset_type.startswith('MSP'):
        fold_list = list(range(0,12,2))
    max_len_unit = str(args.max_len)+'s'
    fold_root = os.path.join(args.test_root, '{}+{}+{}+{}+lr{}+batch_size{}+{}+{}+dropout{}+hidden{}'.format(
        args.model_type, args.dataset_type, max_len_unit, args.feature, args.lr, args.batch_size, 
        args.loss_type, args.optimizer_type, args.dropout, args.hidden
        ))
    if not os.path.exists(os.path.join(fold_root, 'metric_dic_{}.pkl'.format(seed))):
        metric_dic = {'UAR':{},'WAR':{},'UF1':{},'WF1':{}}
    else:
        with open(os.path.join(fold_root, 'metric_dic_{}.pkl'.format(seed)),'rb') as f:
            metric_dic = pickle.load(f)
    metric_dic = main(fold_list, fold_root, metric_dic)
    with open(os.path.join(fold_root, 'metric_dic_{}.pkl'.format(seed)),'wb') as f:
        pickle.dump(metric_dic, f)
