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
import time
from data import Mydataset
from model import FocalLoss, TFN, MulT, TFRNet, MCT, MISA, CHFN, PMR
from utils import mixup_data, mixup_criterion
import argparse
from sklearn.metrics import confusion_matrix,classification_report,recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition")
parser.add_argument('--epochs',default=50,type=int)
parser.add_argument('--batch_size',default=32,type=int)
parser.add_argument('--seed',default=42,type=int)
parser.add_argument('--max_len',default='50', type=str)
parser.add_argument('--lr',default=1e-3,type=float)
parser.add_argument('--hidden',default=128,type=int)
parser.add_argument('--dropout',default=0.25,type=float)
parser.add_argument('--fold',type=int,required=True)
parser.add_argument("--root",type=str,required=True)
parser.add_argument("--feature",type=str,required=True)
parser.add_argument("--dataset_type",type=str,required=True)
parser.add_argument("--loss_type",default='CE',type=str)
parser.add_argument("--optimizer_type",default='AdamW',type=str)
parser.add_argument("--device_number",default='0',type=str)
parser.add_argument("--model_type",type=str,required=True)
parser.add_argument('--missing_rate',default=0.5,type=float)
parser.add_argument('--train_dynamic', default=0,type=int)
args = parser.parse_args()
feature_size_dic = {'fbank':80,'bert':1024,'wavlm':768,'denseface':342}
model_dic = {'TFN':TFN, 'MulT':MulT, 'TFRNet':TFRNet, 'MCT':MCT, 'MISA':MISA, 'CHFN':CHFN, 'PMR':PMR}

def setup_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def count_parameters(model):
    answer = 0
    for p in model.parameters():
        if p.requires_grad:
            answer += p.numel()
            # print(p)
    return answer

def train(model, device, train_loader, criterion, optimizer, epoch, logger, model_name):
    model.train()
    logger.info('start training')
    lr = optimizer.param_groups[0]["lr"]
    logger.info('lr: {:.5f}'.format(lr))
    correct = 0
    for batch, data in tqdm(enumerate(train_loader)):
        audio, audio_mask_padding = data['audio_m'].to(device), data['audio_mask_padding'].to(device)
        vision, vision_mask_padding = data['vision_m'].to(device), data['vision_mask_padding'].to(device)
        text, text_mask_padding = data['text_m'].to(device), data['text_mask_padding'].to(device)
        inputs, inputs_padding_mask = (audio,vision,text),(audio_mask_padding,vision_mask_padding,text_mask_padding)
        label = data['label'].to(device)

        optimizer.zero_grad()
        output = model(inputs, inputs_padding_mask)
        task_loss = criterion(output['label'], label)
        if model_name == 'MISA':
            diff_loss = model.get_diff_loss()
            similarity_loss = model.get_cmd_loss()
            recon_loss = model.get_recon_loss()
            diff_weight = 0.3
            sim_weight = 0.8
            recon_weight = 0.8
            loss = task_loss + \
                    diff_weight * diff_loss + \
                    sim_weight * similarity_loss + \
                    recon_weight * recon_loss            
        elif model_name == 'ICCN':
            similarity_loss = model.get_loss()
            loss = task_loss + 0.5*similarity_loss
        else:
            loss = task_loss
        loss.backward()  
        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], 0.8)
        optimizer.step()

        pred = output['label'].argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
        if batch % 20 == 0:
            logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t loss={:.5f}\t '.format(epoch , batch * len(label), len(train_loader.dataset), 100. * batch / len(train_loader), loss.item()))
            if model_name == 'MISA':
                logger.info('task_loss={:.5f}\t diff_loss={:.5f}\t similarity_loss={:.5f}\t recon_loss={:.5f}\t '.format(task_loss.item(), diff_loss.item(), similarity_loss.item(), recon_loss.item()))
    logger.info('Train set Accuracy: {}/{} ({:.3f}%)'.format(correct, len(train_loader.dataset), 100. * correct / (len(train_loader.dataset))))
    logger.info('finish training!')

def test(model, device, val_loader, criterion, logger, target_names):
    model.eval()
    test_loss = 0
    correct = 0
    logger.info('testing on dev_set')
    
    pred_all = np.array([],dtype=np.long)
    true_all = np.array([],dtype=np.long)

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

    test_loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)

    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), acc))

    con_mat = confusion_matrix(true_all,pred_all)
    cls_rpt = classification_report(true_all,pred_all,target_names=target_names,digits=3)
    logger.info('Confusion Matrix:\n{}\n'.format(con_mat))
    logger.info('Classification Report:\n{}\n'.format(cls_rpt))
    UA = recall_score(true_all,pred_all,average='macro')
    WA = recall_score(true_all,pred_all,average='weighted')
    return test_loss,UA,WA

def early_stopping(network,savepath,metricsInEpochs,gap):
    best_metric_inx=np.argmax(metricsInEpochs)
    if best_metric_inx+1==len(metricsInEpochs):
        best = os.path.join(savepath, 'best_epoch_{}.pt'.format(best_metric_inx+1))
        torch.save(network.state_dict(),best)
        return False
    elif (len(metricsInEpochs)-best_metric_inx >= gap):
        return True
    else:
        return False

def main():
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    seed = args.seed
    max_len = args.max_len
    feature = args.feature
    dataset_type = args.dataset_type
    root = args.root
    fold = args.fold
    loss_type = args.loss_type
    optimizer_type = args.optimizer_type
    device_number = args.device_number
    model_type = args.model_type
    missing_rate = args.missing_rate
    train_dynamic = args.train_dynamic

    os.environ["CUDA_VISIBLE_DEVICES"] = device_number
    setup_seed(seed)
    savedir = os.path.join(root, 'fold_{}'.format(fold))
    try:
        os.makedirs(savedir)
    except OSError:
        if not os.path.isdir(savedir):
            raise

    subprocess.check_call(["cp", "model.py", savedir])
    subprocess.check_call(["cp", "train.py", savedir])
    subprocess.check_call(["cp", "data.py", savedir])
    subprocess.check_call(["cp", "utils.py", savedir])
    subprocess.check_call(["cp", "run.sh", savedir])
    logpath = savedir + "/exp.log"
    modelpath = savedir + "/model.pt"
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    feature_ls = feature.split(',')   
    max_len_ls = [int(x) for x in max_len.split(',')]

    if '_' in model_type:
        model_name, model_layer = model_type.split('_')
        model_layer = int(model_layer)
    else:
        model_name = model_type
        model_layer = 4

    train_set = Mydataset(dataset_type=dataset_type, mode='train', max_len_ls=max_len_ls, fold=fold, feature_ls=feature_ls, missing_rate=missing_rate, seed=seed, train_dynamic=train_dynamic) 
    dev_set = Mydataset(dataset_type=dataset_type, mode='val', max_len_ls=max_len_ls, fold=fold, feature_ls=feature_ls, missing_rate=missing_rate, seed=seed)
    val_set = Mydataset(dataset_type=dataset_type, mode='test', max_len_ls=max_len_ls, fold=fold, feature_ls=feature_ls, missing_rate=missing_rate, seed=seed)

    drop_last = True if len(train_set)%batch_size<8 else False
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=drop_last)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    logger = get_logger(logpath)
    logger.info(args)
    logger.info('train_set speaker names: {}'.format(train_set.SpeakerNames))
    logger.info('val_set speaker names: {}'.format(dev_set.SpeakerNames))
    logger.info('test speaker names: {}'.format(val_set.SpeakerNames))

    model = model_dic[model_name](feature_size=[feature_size_dic[feature] for feature in feature_ls], emotion_cls=train_set.NumClasses, h_dims=args.hidden, dropout=args.dropout, layers=model_layer).to(device)
    logger.info('The model {} has {} trainable parameters.'.format(model_type, count_parameters(model)))

    if loss_type == 'CE':
        #criterion = nn.NLLLoss(weight=train_set.weight.to(device))
        criterion = nn.NLLLoss()
        criterion_test = nn.NLLLoss(reduction='sum')
    elif loss_type == 'Focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.)
        criterion_test = FocalLoss(alpha=0.25, gamma=2., reduction='sum')
    else:
        raise NameError

    if optimizer_type == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        raise NameError
    
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    val_UA_list = []
    test_UA_dic = {}
    test_WA_dic = {}

    for epoch in range(1, epochs+1):
        start = time.time()
        train(model, device, train_loader, criterion, optimizer, epoch, logger, model_name)   
        time.sleep(0.003)
        val_loss,val_UA,_ = test(model, device, dev_loader, criterion_test, logger, train_set.ClassNames)
        time.sleep(0.003)
        test_loss,test_UA,test_WA = test(model, device, val_loader, criterion_test, logger, train_set.ClassNames)
        end = time.time()
        duration = end-start
        val_UA_list.append(val_UA)
        if early_stopping(model,savedir,val_UA_list,gap=10):
            break
        test_UA_dic[test_UA] = epoch
        test_WA_dic[test_WA] = epoch
        #scheduler.step(val_UA)
        logger.info("-"*50)
        logger.info('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        logger.info("-"*50)
        time.sleep(0.003)

    best_UA=max(test_UA_dic.keys())
    best_WA=max(test_WA_dic.keys())
    logger.info('UA dic: {}'.format(test_UA_dic))
    logger.info('WA dic: {}'.format(test_WA_dic))    
    logger.info('best UA: {}  @epoch: {}'.format(best_UA,test_UA_dic[best_UA]))
    logger.info('best WA: {}  @epoch: {}'.format(best_WA,test_WA_dic[best_WA]))   
    torch.save(model.state_dict(), modelpath)

    if int(fold)==11:
        sys.stdout.flush()
        input('[Press Any Key to start another run]')
if __name__ == '__main__':
    main()
