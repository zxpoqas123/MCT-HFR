import math
from typing import Callable, Optional
import numpy as np
import random
import librosa

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
#from torchaudio import functional as audioF
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Function
from collections import OrderedDict
from modules.transformer import TransformerEncoder, MITransformerEncoder
from modules.mctransformer import MCTransformerEncoder
from torch.nn.init import xavier_normal

class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2, weight = None, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.nllloss = nn.NLLLoss(weight = weight, reduction = 'none')

    def forward(self, output, label):
        logp = self.nllloss(output, label)
        p = torch.exp(-logp)
        loss = self.alpha * (1-p)**self.gamma * logp
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

######################################## LF ########################################
class VTE(nn.Module):
    def __init__(self, feature_size=1024, emotion_cls=4, h_dims=128, dropout=0.25):
        super(VTE, self).__init__()
        #self.ln_a = nn.LayerNorm(feature_size[0])
        #self.ln_t = nn.LayerNorm(feature_size[1])

        #self.RNN_a = nn.GRU(input_size=feature_size[0],hidden_size=h_dims,num_layers=2,batch_first=True,dropout=dropout,bidirectional=True)
        #self.RNN_t = nn.GRU(input_size=feature_size[1],hidden_size=h_dims,num_layers=2,batch_first=True,dropout=dropout,bidirectional=True)
        self.inp0 = nn.Conv1d(feature_size[0], h_dims, kernel_size=1, padding=0, bias=False)       
        self.vte = self.get_network(embed_dim=h_dims, layers=1, attn_dropout=dropout)

        self.EmotionClassifier = nn.Sequential(
                nn.Linear(h_dims, emotion_cls),
                nn.LogSoftmax(dim=-1))

    def get_network(self, embed_dim, layers, attn_dropout):
        self.num_heads = 4
        self.relu_dropout = 0.2
        self.res_dropout = 0.2
        self.attn_mask = True
        return TransformerEncoder(embed_dim=embed_dim, 
                                  layers=layers,
                                  attn_dropout=attn_dropout,
                                  num_heads=self.num_heads,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, inputs):
        H,lens = pad_packed_sequence(inputs,batch_first=True)
        H = self.inp0(H.transpose(1, 2)).permute(2, 0, 1)
        div = torch.unsqueeze(lens.float(), 1).to(H.device)

        H = self.vte(H)
        mask = (torch.arange(H.shape[0])[:, None] < lens[None, :]).unsqueeze(2).to(H.device)
        emo_ebd = (H*mask).sum(0)/div
        pred = self.EmotionClassifier(emo_ebd)

        return pred, emo_ebd

class LF(nn.Module):
    def __init__(self, feature_size=1024, emotion_cls=4, h_dims=128, dropout=0.25):
        super(LF, self).__init__()
        self.h_dims = h_dims
        self.inp0 = nn.Conv1d(feature_size[0], self.h_dims, kernel_size=1, padding=0, bias=False)
        self.inp1 = nn.Conv1d(feature_size[1], self.h_dims, kernel_size=1, padding=0, bias=False)
        self.inp2 = nn.Conv1d(feature_size[2], self.h_dims, kernel_size=1, padding=0, bias=False)       
        self.vte0 = self.get_network(embed_dim=self.h_dims, layers=1, attn_dropout=dropout)
        self.vte1 = self.get_network(embed_dim=self.h_dims, layers=1, attn_dropout=dropout)
        self.vte2 = self.get_network(embed_dim=self.h_dims, layers=1, attn_dropout=dropout)

        self.EmotionClassifier = nn.Sequential(
                nn.Linear(self.h_dims*3, emotion_cls),
                nn.LogSoftmax(dim=-1))

    def get_network(self, embed_dim, layers, attn_dropout):
        self.num_heads = 4
        self.relu_dropout = 0.2
        self.res_dropout = 0.2
        self.attn_mask = True
        return TransformerEncoder(embed_dim=embed_dim,
                                  layers=layers,
                                  attn_dropout=attn_dropout,
                                  num_heads=self.num_heads,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  attn_mask=self.attn_mask)

    def cal_intrinsics(self, H, lens, vte):
        div = torch.unsqueeze(lens.float(), 1).to(H.device)
        H = vte(H)
        mask = (torch.arange(H.shape[0])[:, None] < lens[None, :]).unsqueeze(2).to(H.device)
        emo_ebd = (H*mask).sum(0)/div
        return emo_ebd

    def forward(self, inputs):
        H0,lens0 = pad_packed_sequence(inputs[0],batch_first=True)
        H0 = self.inp0(H0.transpose(1, 2)).permute(2, 0, 1)

        H1,lens1 = pad_packed_sequence(inputs[1],batch_first=True)
        H1 = self.inp1(H1.transpose(1, 2)).permute(2, 0, 1)

        H2,lens2 = pad_packed_sequence(inputs[2],batch_first=True)
        H2 = self.inp2(H2.transpose(1, 2)).permute(2, 0, 1)        

        emo_ebd0= self.cal_intrinsics(H0, lens0, self.vte0)
        emo_ebd1= self.cal_intrinsics(H1, lens1, self.vte1)
        emo_ebd2= self.cal_intrinsics(H2, lens2, self.vte2)   

        emo_ebd = torch.cat([emo_ebd0,emo_ebd1,emo_ebd2],-1)
        pred = self.EmotionClassifier(emo_ebd)

        return pred, emo_ebd

class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.LayerNorm(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        #self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, x_mask):
        '''
        Args:
            x: tensor of shape (batch, src_len, embed_dim)
            x_mask: tensor of shape (batch, src_len)
        '''
        normed = self.norm(x) if x.shape[0]>1 else x
        x_transformed = self.linear_1(normed)
        x_mask = torch.logical_not(x_mask)
        x = x_transformed.sum(1)/(x_mask.sum(1).unsqueeze(1))
        #y_1 = self.drop(F.relu(x))
        #y_2 = self.linear_2(y_1)
        return x, x_transformed

class VisionSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for Vision
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(VisionSubNet, self).__init__()
        if num_layers == 1:
            dropout = 0.0
        self.n_directions = 2 if bidirectional else 1
        self.d_out = hidden_size
        self.n_layers = num_layers
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(self.n_directions*hidden_size, out_size)

    def forward(self, x, x_mask):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        x_mask = torch.logical_not(x_mask)
        max_len = x_mask.shape[1]
        x_lens = x_mask.sum(1)
        x_packed = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        x_hidden, final_states = self.rnn(x_packed)
        h_n = final_states[0] # (n_directions*n_layer, batch_size, dim)
        batch_size = x.shape[0]
        h_n = h_n.view(self.n_layers, self.n_directions, batch_size, self.d_out) # (n_layer, n_directions, batch_size, dim)
        last_layer = h_n[-1].permute(1,0,2) # (BS, ND, dim)
        x_out = last_layer.view(batch_size, self.n_directions * self.d_out) # (BS, ND*dim)
        h = self.dropout(x_out)
        y_1 = self.linear_1(h)
        hidden_states = pad_packed_sequence(x_hidden,batch_first=True,total_length=max_len)[0]
        return y_1, hidden_states

######################################## TFN ########################################
#input: (batch, src_len, embed_dim)
#(A,V,L)
class TFN(nn.Module):
    def __init__(self, feature_size=[512,512,512], emotion_cls=4, h_dims=128, dropout=0.25, layers=4):
        super(TFN, self).__init__()
        self.audio_in, self.video_in, self.text_in = feature_size
        self.audio_hidden, self.video_hidden, self.text_hidden= [h_dims]*3
        self.output_dim = emotion_cls
        self.post_fusion_dim = 64

        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = [dropout]*4

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, dropout=self.audio_prob)
        self.video_subnet = VisionSubNet(self.video_in, self.video_hidden, self.video_hidden, num_layers=layers, dropout=self.video_prob)
        self.text_subnet = SubNet(self.text_in, self.text_hidden, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_hidden + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Sequential(
            nn.Linear(self.post_fusion_dim, self.output_dim),
            nn.LogSoftmax(dim=-1))

    def forward(self, inputs, input_padding_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        audio,vision,text = inputs
        audio_mask_padding,vision_mask_padding,text_mask_padding = input_padding_masks

        audio_h,_ = self.audio_subnet(audio, audio_mask_padding)
        video_h,_ = self.video_subnet(vision, vision_mask_padding)
        text_h,_ = self.text_subnet(text, text_mask_padding)
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        add_one = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(audio_h).to(text.device)
        _audio_h = torch.cat((add_one, audio_h), dim=1)
        _video_h = torch.cat((add_one, video_h), dim=1)
        _text_h = torch.cat((add_one, text_h), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        
        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped), inplace=True)
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1), inplace=True)
        output = self.post_fusion_layer_3(post_fusion_y_2)

        res = {'label':output, 'embed':fusion_tensor}
        return res

######################################## LMF ########################################
#input: (batch, src_len, embed_dim)
#(A,V,L)
class LMF(nn.Module):
    def __init__(self, feature_size=[512,512,512], emotion_cls=4, h_dims=128, dropout=0.25, layers=4):
        super(LMF, self).__init__()
        self.audio_in, self.video_in, self.text_in = feature_size
        self.audio_hidden, self.video_hidden, self.text_hidden= [h_dims]*3
        self.output_dim = emotion_cls
        self.rank = 16
        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = [dropout]*4

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, dropout=self.audio_prob)
        self.video_subnet = VisionSubNet(self.video_in, self.video_hidden, self.video_hidden, num_layers=layers, dropout=self.video_prob)
        self.text_subnet = SubNet(self.text_in, self.text_hidden, dropout=self.text_prob)

        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_hidden + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_hidden + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, self.text_hidden + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # define the post_fusion layers
        self.post_fusion_layer_3 = nn.LogSoftmax(dim=-1)

        # init teh factors
        xavier_normal(self.audio_factor)
        xavier_normal(self.video_factor)
        xavier_normal(self.text_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, inputs, input_padding_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        audio,vision,text = inputs
        audio_mask_padding,vision_mask_padding,text_mask_padding = input_padding_masks

        audio_h,_ = self.audio_subnet(audio, audio_mask_padding)
        video_h,_ = self.video_subnet(vision, vision_mask_padding)
        text_h,_ = self.text_subnet(text, text_mask_padding)
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        add_one = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(audio_h).to(text.device)
        _audio_h = torch.cat((add_one, audio_h), dim=1)
        _video_h = torch.cat((add_one, video_h), dim=1)
        _text_h = torch.cat((add_one, text_h), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        
        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = fusion_tensor.view(-1, self.output_dim)
        output = self.post_fusion_layer_3(output)

        res = {'label':output, 'embed':fusion_tensor}
        return res

######################################## ICCN ########################################
#input: (batch, src_len, embed_dim)
#(A,V,L)
class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, learn_bn, use_relu):
        super(conv_layer, self).__init__()
        self.use_relu = use_relu
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, 3, 1, 1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(out_channels//2, affine=learn_bn)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, 3, 1, 1, bias=False)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.bn2 = nn.BatchNorm2d(out_channels, affine=learn_bn)
        self.maxpool = nn.MaxPool2d(2, 2, 0)
        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.use_relu:
            out = F.relu(out)
        out = self.maxpool(out)
        out = self.bn2(self.conv2(out))
        if self.use_relu:
            out = F.relu(out)
        avg_pool = self.globalavgpool(out)
        avg_pool = avg_pool.view(x.shape[0], -1)
        return avg_pool

class CCALoss(nn.Module):
    def __init__(self, outdim_size, use_all_singular_values):
        super(CCALoss, self).__init__()
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values

    def forward(self, H1, H2):
        device = H1.device
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        o1 = o2 = H1.size(0)
        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=device)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(device)) # regularization for more stability
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U>eps, U, (torch.ones(U.shape).float()*eps).to(device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr

class ICCN(nn.Module):
    def __init__(self, feature_size=[512,512,512], emotion_cls=4, h_dims=128, dropout=0.25, layers=4):
        super(ICCN, self).__init__()
        self.audio_in, self.video_in, self.text_in = feature_size
        self.audio_hidden, self.video_hidden, self.text_hidden= [h_dims]*3
        self.output_dim = emotion_cls

        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = [dropout]*4

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, dropout=self.audio_prob)
        self.video_subnet = VisionSubNet(self.video_in, self.video_hidden, self.video_hidden, num_layers=layers, dropout=self.video_prob)
        self.text_subnet = SubNet(self.text_in, self.text_hidden, dropout=self.text_prob)

        self.cnn_at = conv_layer(in_channels=1, out_channels=self.text_hidden, learn_bn=False, use_relu=True)
        self.cnn_vt = conv_layer(in_channels=1, out_channels=self.text_hidden, learn_bn=False, use_relu=True)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_hidden*3, self.text_hidden)
        self.post_fusion_layer_2 = nn.Sequential(
            nn.Linear(self.text_hidden, self.output_dim),
            nn.LogSoftmax(dim=-1))
        #self.loss = CMD()
        self.loss = CCALoss(outdim_size=8, use_all_singular_values=False)

    def get_loss(self,):
        #loss = self.loss(self.at_h, self.vt_h, 5)
        loss = self.loss(self.at_h.permute(1,0), self.vt_h.permute(1,0))
        return loss

    def forward(self, inputs, input_padding_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        audio,vision,text = inputs
        audio_mask_padding,vision_mask_padding,text_mask_padding = input_padding_masks

        audio_h,_ = self.audio_subnet(audio, audio_mask_padding)
        video_h,_ = self.video_subnet(vision, vision_mask_padding)
        text_h,_ = self.text_subnet(text, text_mask_padding)
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        at_fusion = torch.bmm(audio_h.unsqueeze(2), text_h.unsqueeze(1)).unsqueeze(1)
        vt_fusion = torch.bmm(video_h.unsqueeze(2), text_h.unsqueeze(1)).unsqueeze(1)

        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it

        at_h = self.cnn_at(at_fusion)
        vt_h = self.cnn_vt(vt_fusion)
        self.at_h, self.vt_h = at_h, vt_h
        
        fusion_tensor = torch.cat([text_h, at_h, vt_h],-1)
        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped), inplace=True)
        output = self.post_fusion_layer_2(post_fusion_y_1)

        res = {'label':output, 'embed':fusion_tensor}
        return res

######################################## MulT ########################################
#input: (batch_size, seq_len, n_features)
#(A,V,L)
class MulT(nn.Module):
    def __init__(self, feature_size=[512,512,512], emotion_cls=4, h_dims=128, dropout=0.25, layers=4):
        """
        Construct a MulT model.
        """
        super(MulT, self).__init__()
        self.orig_d_a, self.orig_d_v, self.orig_d_l = feature_size
        self.d_a, self.d_v, self.d_l = [h_dims]*3
        self.num_heads = 4
        self.layers = layers
        self.attn_dropout = 0.0
        self.attn_dropout_a = 0.0
        self.attn_dropout_v = 0.0
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = dropout
        self.attn_mask = False

        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la', layers=self.layers)
        self.trans_l_with_v = self.get_network(self_type='lv', layers=self.layers)
        
        self.trans_a_with_l = self.get_network(self_type='al', layers=self.layers)
        self.trans_a_with_v = self.get_network(self_type='av', layers=self.layers)
        
        self.trans_v_with_l = self.get_network(self_type='vl', layers=self.layers)
        self.trans_v_with_a = self.get_network(self_type='va', layers=self.layers)
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=2)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=2)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=2)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Sequential(
            nn.Linear(combined_dim, emotion_cls),
            nn.LogSoftmax(dim=-1))

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask, 
                                  position_embedding=True)

    def masked_avg_pooling(self, x, x_mask_padding):
        # x: Dimension (seq_len, batch_size, n_features)
        # x_mask_padding: Dimension (batch_size, seq_len)
        x_mask_values = torch.logical_not(x_mask_padding).transpose(0,1).unsqueeze(-1)  # True indicates the position of values, Dimension (seq_len,batch_size,1)
        x_avg_pool = ((x*x_mask_values).sum(0))/(x_mask_values.sum(0))  # x_avg_pool: (batch_size, n_features)
        return x_avg_pool

    def forward(self, inputs, input_padding_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        key_padding_mask_a, key_padding_mask_v, and key_padding_mask_l should have dimension [batch_size, seq_len]
        """
        audio,vision,text = inputs
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l = input_padding_masks
        x_a, x_v, x_l = audio.transpose(1, 2), vision.transpose(1, 2), text.transpose(1, 2)
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a, key_padding_mask_a)    # Dimension (seq_len, batch_size, n_features)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v, key_padding_mask_v)    # Dimension (seq_len, batch_size, n_features)
        h_ls = torch.cat([h_l_with_as[0], h_l_with_vs[0]], dim=2)
        h_ls_final = self.trans_l_mem(h_ls, key_padding_mask=key_padding_mask_l)   #Transformer output (x, x_intermediates, attn_intermediates)
        last_h_l = self.masked_avg_pooling(h_ls_final[0], key_padding_mask_l)   # Take the masked averaged output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l, key_padding_mask_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v, key_padding_mask_v)
        h_as = torch.cat([h_a_with_ls[0], h_a_with_vs[0]], dim=2)
        h_as_final = self.trans_a_mem(h_as, key_padding_mask=key_padding_mask_a)
        last_h_a = self.masked_avg_pooling(h_as_final[0], key_padding_mask_a)

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l, key_padding_mask_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a, key_padding_mask_a)
        h_vs = torch.cat([h_v_with_ls[0], h_v_with_as[0]], dim=2)
        h_vs_final = self.trans_v_mem(h_vs, key_padding_mask=key_padding_mask_v)
        last_h_v = self.masked_avg_pooling(h_vs_final[0], key_padding_mask_v)
        
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)

        res = {'label':output, 'embed':last_hs}
        return res

######################################## TFR_Net ########################################        
#input: (batch_size, seq_len, n_features)
#(A,V,L)
class GRUencoder(nn.Module):
    """Pad for utterances with variable lengths and maintain the order of them after GRU"""
    def __init__(self, embedding_dim, utterance_dim, num_layers):
        super(GRUencoder, self).__init__()
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=utterance_dim,
                          bidirectional=True, num_layers=num_layers)

    def forward(self, utterance, utterance_lens):
        """Server as simple GRU Layer.
        Args:
            utterance (tensor): [utter_num, max_word_len, embedding_dim]
            utterance_lens (tensor): [utter_num]
        Returns:
            transformed utterance representation (tensor): [utter_num, max_word_len, 2 * utterance_dim]
        """
        utterance_embs = utterance.transpose(0,1)
    
        # SORT BY LENGTH.
        sorted_utter_length, indices = torch.sort(utterance_lens, descending=True)
        _, indices_unsort = torch.sort(indices)
        
        s_embs = utterance_embs.index_select(1, indices)

        # PADDING & GRU MODULE & UNPACK.
        utterance_packed = pack_padded_sequence(s_embs, sorted_utter_length.cpu())
        utterance_output = self.gru(utterance_packed)[0]
        utterance_output = pad_packed_sequence(utterance_output, total_length=utterance.size(1))[0]

        # UNSORT BY LENGTH.
        utterance_output = utterance_output.index_select(1, indices_unsort)
        return utterance_output.transpose(0,1)

class C_GATE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, drop):
        super(C_GATE, self).__init__()

        # BI-GRU to get the historical context.
        self.gru = GRUencoder(embedding_dim, hidden_dim, num_layers)
        # Calculate the gate.
        self.cnn = nn.Conv1d(in_channels= 2 * hidden_dim, out_channels=1, kernel_size=3, stride=1, padding=1)
        # Linear Layer to get the representation.
        self.fc = nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim)
        # Utterance Dropout.
        self.dropout_in = nn.Dropout(drop)
        
    def forward(self, utterance, utterance_mask):
        """Returns:
            utterance_rep: [utter_num, utterance_dim]
        """
        utterance_mask = torch.logical_not(utterance_mask)
        utterance_lens = utterance_mask.sum(1)

        # Bi-GRU
        transformed_ = self.gru(utterance, utterance_lens) # [batch_size, seq_len, 2 * hidden_dim]
        # CNN_GATE MODULE.
        gate = F.sigmoid(self.cnn(transformed_.transpose(1, 2)).transpose(1, 2))  # [batch_size, seq_len, 1]
        # CALCULATE GATE OUTPUT.
        gate_x = torch.tanh(transformed_) * gate # [batch_size, seq_len, 2 * hidden_dim]
        # SPACE TRANSFORMS
        utterance_rep = torch.tanh(self.fc(torch.cat([utterance, gate_x], dim=-1))) # [batch_size, seq_len, hidden_dim]
        # MAXPOOLING LAYERS
        utterance_rep = torch.max(utterance_rep, dim=1)[0] # [batch_size, hidden_dim]
        # UTTERANCE DROPOUT
        utterance_rep = self.dropout_in(utterance_rep) # [utter_num, utterance_dim]
        return utterance_rep

class GATE_F(nn.Module):
    def __init__(self, emotion_cls=4, embedding_dim=384, h_dims=128, dropout=0.25):
        super(GATE_F, self).__init__()
        
        self.text_encoder = C_GATE(embedding_dim, h_dims, 2, dropout)
        self.audio_encoder = C_GATE(embedding_dim, h_dims, 2, dropout)
        self.vision_encoder = C_GATE(embedding_dim, h_dims, 2, dropout)
        self.norm = nn.BatchNorm1d(h_dims*3)
        # Classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module('linear_trans_hidden', nn.Linear(h_dims*3, h_dims*2))
        self.classifier.add_module('linear_trans_activation', nn.LeakyReLU())
        self.classifier.add_module('linear_trans_drop', nn.Dropout(0.1))
        self.classifier.add_module('linear_trans_final', nn.Linear(h_dims*2, emotion_cls))
        self.classifier.add_module('softmax', nn.LogSoftmax(dim=-1))

    def forward(self, inputs, input_padding_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        key_padding_mask_a, key_padding_mask_v, and key_padding_mask_l should have dimension [batch_size, seq_len]
        """
        audio,vision,text = inputs
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l = input_padding_masks

        text_rep = self.text_encoder(text, key_padding_mask_l)
        audio_rep = self.audio_encoder(audio, key_padding_mask_a)
        vision_rep = self.vision_encoder(vision, key_padding_mask_v)

        utterance_rep = torch.cat((text_rep, audio_rep, vision_rep), dim=1)
        utterance_normed = self.norm(utterance_rep) if utterance_rep.shape[0]>1 else utterance_rep
        output = self.classifier(utterance_normed)
        res = {'label':output, 'embed':utterance_rep}
        return res

class GATE_F_uni(nn.Module):
    def __init__(self, emotion_cls=4, embedding_dim=384, h_dims=128, dropout=0.25):
        super(GATE_F_uni, self).__init__()
        
        self.encoder = C_GATE(embedding_dim, h_dims, 2, dropout)
        self.norm = nn.BatchNorm1d(h_dims)
        # Classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module('linear_trans_hidden', nn.Linear(h_dims, h_dims*2))
        self.classifier.add_module('linear_trans_activation', nn.LeakyReLU())
        self.classifier.add_module('linear_trans_drop', nn.Dropout(0.1))
        self.classifier.add_module('linear_trans_final', nn.Linear(h_dims*2, emotion_cls))
        self.classifier.add_module('softmax', nn.LogSoftmax(dim=-1))

    def forward(self, inputs, input_padding_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        key_padding_mask_a, key_padding_mask_v, and key_padding_mask_l should have dimension [batch_size, seq_len]
        """
        utterance_rep = self.encoder(inputs, input_padding_masks)
        utterance_normed = self.norm(utterance_rep) if utterance_rep.shape[0]>1 else utterance_rep
        output = self.classifier(utterance_normed)
        res = {'label':output, 'embed':utterance_rep}
        return res

class TFRNet(nn.Module):
    def __init__(self, feature_size=[512,512,512], emotion_cls=4, h_dims=128, dropout=0.25, layers=4):
        super(TFRNet, self).__init__()
        self.orig_d_a, self.orig_d_v, self.orig_d_l = feature_size
        self.d_a, self.d_v, self.d_l = [h_dims]*3
        self.num_heads = 4
        self.layers = layers
        self.attn_dropout = 0.0
        self.attn_dropout_a = 0.0
        self.attn_dropout_v = 0.0
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = dropout
        self.attn_mask = False

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la', layers=self.layers)
        self.trans_l_with_v = self.get_network(self_type='lv', layers=self.layers)
        
        self.trans_a_with_l = self.get_network(self_type='al', layers=self.layers)
        self.trans_a_with_v = self.get_network(self_type='av', layers=self.layers)
        
        self.trans_v_with_l = self.get_network(self_type='vl', layers=self.layers)
        self.trans_v_with_a = self.get_network(self_type='va', layers=self.layers)
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l = self.get_network(self_type='l', layers=self.layers)
        self.trans_a = self.get_network(self_type='a', layers=self.layers)
        self.trans_v = self.get_network(self_type='v', layers=self.layers)
       
        # Projection layers
        self.fusion_subnet = GATE_F(emotion_cls=emotion_cls, embedding_dim=3*h_dims, h_dims=h_dims, dropout=dropout)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask, 
                                  position_embedding=True)

    def forward(self, inputs, input_padding_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        key_padding_mask_a, key_padding_mask_v, and key_padding_mask_l should have dimension [batch_size, seq_len]
        """
        audio, vision, text = inputs
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l = input_padding_masks
        x_a, x_v, x_l = audio.transpose(1, 2), vision.transpose(1, 2), text.transpose(1, 2)
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # alignment
        # (V,A) --> L
        h_l = self.trans_l(proj_x_l, key_padding_mask=key_padding_mask_l)
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a, key_padding_mask_a)    # Dimension (seq_len, batch_size, n_features)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v, key_padding_mask_v)    # Dimension (seq_len, batch_size, n_features)
        h_ls = torch.cat([h_l[0], h_l_with_as[0], h_l_with_vs[0]], dim=2)  #Transformer output (x, x_intermediates, attn_intermediates)
        h_ls = h_ls.transpose(0, 1)   # shape: (L,B,D) -> (B,L,D)

        # (L,V) --> A
        h_a = self.trans_a(proj_x_a, key_padding_mask=key_padding_mask_a)
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l, key_padding_mask_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v, key_padding_mask_v)
        h_as = torch.cat([h_a[0], h_a_with_ls[0], h_a_with_vs[0]], dim=2)
        h_as = h_as.transpose(0, 1)

        # (L,A) --> V
        h_v = self.trans_v(proj_x_v, key_padding_mask=key_padding_mask_v)
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l, key_padding_mask_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a, key_padding_mask_a)
        h_vs = torch.cat([h_v[0], h_v_with_ls[0], h_v_with_as[0]], dim=2)
        h_vs = h_vs.transpose(0, 1)

        # fusion&classification
        res = self.fusion_subnet((h_as,h_vs,h_ls),input_padding_masks)
        return res

######################################## CHFN ########################################        
#input: (batch_size, seq_len, n_features)
#(A,V,L)
class CHFN(nn.Module):
    def __init__(self, feature_size=[512,512,512], emotion_cls=4, h_dims=128, dropout=0.25, layers=4):
        super(CHFN, self).__init__()
        self.orig_d_a, self.orig_d_v, self.orig_d_l = feature_size
        self.d_a, self.d_v, self.d_l = [h_dims]*3
        self.num_heads = 4
        self.layers = layers
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = dropout

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Multimodal interaction attentions
        self.mit = MITransformerEncoder(embed_dim=h_dims,
                                        num_heads=self.num_heads,
                                        layers=layers,
                                        relu_dropout=self.relu_dropout,
                                        res_dropout=self.res_dropout,
                                        embed_dropout=self.embed_dropout,
                                        position_embedding=True,
                                        beta_shift=1.0)                                                                                   
        # Projection layers
        self.classification = GATE_F_uni(emotion_cls=emotion_cls, embedding_dim=h_dims, h_dims=h_dims//2, dropout=dropout)

    def forward(self, inputs, input_padding_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        key_padding_mask_a, key_padding_mask_v, and key_padding_mask_l should have dimension [batch_size, seq_len]
        """
        audio, vision, text = inputs
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l = input_padding_masks
        x_a, x_v, x_l = audio.transpose(1, 2), vision.transpose(1, 2), text.transpose(1, 2)
        # Project the textual/visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        proj_nonverbal = torch.cat([proj_x_a,proj_x_v],dim=0)
        key_padding_mask_nonverbal = torch.cat([key_padding_mask_a, key_padding_mask_v],dim=1)

        # alignment
        h_x = self.mit(proj_x_l, proj_nonverbal, proj_nonverbal, key_padding_mask_nonverbal)
        h_x = h_x[0].transpose(0, 1)        # shape: (L,B,D) -> (B,L,D)

        # fusion&classification
        res = self.classification(h_x, key_padding_mask_l)
        return res

######################################## PMR ########################################        
#input: (batch_size, seq_len, n_features)
#(A,V,L)
class PMR(nn.Module):
    def __init__(self, feature_size=[512,512,512], emotion_cls=4, h_dims=128, dropout=0.25, layers=4):
        super(PMR, self).__init__()
        self.orig_d_a, self.orig_d_v, self.orig_d_l = feature_size
        self.d_a, self.d_v, self.d_l = [h_dims]*3
        self.num_heads = 4
        self.layers = layers
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = dropout

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Multimodal interaction attentions
        self.mit_a = MITransformerEncoder(embed_dim=h_dims,
                                        num_heads=self.num_heads,
                                        layers=layers,
                                        relu_dropout=self.relu_dropout,
                                        res_dropout=self.res_dropout,
                                        embed_dropout=self.embed_dropout,
                                        position_embedding=True,
                                        beta_shift=1.0)                                                                                   

        self.mit_v = MITransformerEncoder(embed_dim=h_dims,
                                        num_heads=self.num_heads,
                                        layers=layers,
                                        relu_dropout=self.relu_dropout,
                                        res_dropout=self.res_dropout,
                                        embed_dropout=self.embed_dropout,
                                        position_embedding=True,
                                        beta_shift=1.0) 

        self.mit_l = MITransformerEncoder(embed_dim=h_dims,
                                        num_heads=self.num_heads,
                                        layers=layers,
                                        relu_dropout=self.relu_dropout,
                                        res_dropout=self.res_dropout,
                                        embed_dropout=self.embed_dropout,
                                        position_embedding=True,
                                        beta_shift=1.0) 

        # Projection layers
        self.fusion_subnet = GATE_F(emotion_cls=emotion_cls, embedding_dim=h_dims, h_dims=h_dims//2, dropout=dropout)

    def forward(self, inputs, input_padding_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        key_padding_mask_a, key_padding_mask_v, and key_padding_mask_l should have dimension [batch_size, seq_len]
        """
        audio, vision, text = inputs
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l = input_padding_masks
        x_a, x_v, x_l = audio.transpose(1, 2), vision.transpose(1, 2), text.transpose(1, 2)
        # Project the textual/visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        proj_vl = torch.cat([proj_x_v,proj_x_l],dim=0)
        key_padding_mask_vl = torch.cat([key_padding_mask_v, key_padding_mask_l],dim=1)
        h_a = self.mit_a(proj_x_a, proj_vl, proj_vl, key_padding_mask_vl)
        h_as = h_a[0].transpose(0, 1)        # shape: (L,B,D) -> (B,L,D)

        proj_la = torch.cat([proj_x_l,proj_x_a],dim=0)
        key_padding_mask_la = torch.cat([key_padding_mask_l, key_padding_mask_a],dim=1)
        h_v = self.mit_v(proj_x_v, proj_la, proj_la, key_padding_mask_la)
        h_vs = h_v[0].transpose(0, 1)        # shape: (L,B,D) -> (B,L,D)

        proj_av = torch.cat([proj_x_a,proj_x_v],dim=0)
        key_padding_mask_av = torch.cat([key_padding_mask_a, key_padding_mask_v],dim=1)
        h_l = self.mit_l(proj_x_l, proj_av, proj_av, key_padding_mask_av)
        h_ls = h_l[0].transpose(0, 1)        # shape: (L,B,D) -> (B,L,D)

        res = self.fusion_subnet((h_as,h_vs,h_ls),input_padding_masks)
        return res

        
######################################## MCT ########################################
#input: (batch_size, seq_len, n_features)
#(A,V,L)
class MCT(nn.Module):
    def __init__(self, feature_size=[512,512,512], emotion_cls=4, h_dims=128, dropout=0.25, layers=4):
        super(MCT, self).__init__()
        self.orig_d_a, self.orig_d_v, self.orig_d_l = feature_size
        self.d = h_dims
        self.num_heads = 4
        self.layers = layers
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = dropout
        
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d, kernel_size=1, padding=0, bias=False)

        # 2. Multimodal collaborative attentions
        self.mct = MCTransformerEncoder(embed_dim=h_dims,
                                        num_heads=self.num_heads,
                                        layers=layers,
                                        relu_dropout=self.relu_dropout,
                                        res_dropout=self.res_dropout,
                                        embed_dropout=self.embed_dropout,
                                        position_embedding=True)       
        # Projection layers
        self.fusion_subnet = GATE_F(emotion_cls=emotion_cls, embedding_dim=h_dims, h_dims=h_dims//2, dropout=dropout)

    def conv_proj(self, audio, vision, text):
        x_a, x_v, x_l = audio.transpose(1, 2), vision.transpose(1, 2), text.transpose(1, 2)
        proj_x_a = x_a if self.orig_d_a == self.d else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d else self.proj_v(x_v)
        proj_x_l = x_l if self.orig_d_l == self.d else self.proj_l(x_l)        
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        return proj_x_a, proj_x_v, proj_x_l

    def forward(self, inputs, input_padding_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        key_padding_mask_a, key_padding_mask_v, and key_padding_mask_l should have dimension [batch_size, seq_len]
        """
        audio, vision, text = inputs
        # Project the textual/visual/audio features
        proj_x_a, proj_x_v, proj_x_l = self.conv_proj(audio, vision, text)
        # alignment
        (h_as,h_vs,h_ls),_,_ = self.mct((proj_x_a,proj_x_v,proj_x_l), input_padding_masks)
        h_as,h_vs,h_ls = h_as.transpose(0, 1),h_vs.transpose(0, 1), h_ls.transpose(0, 1)   # shape: (L,B,D) -> (B,L,D)
        # fusion&classification
        res = self.fusion_subnet((h_as,h_vs,h_ls),input_padding_masks)
        return res

######################################## MISA ########################################
#input: (batch_size, seq_len, n_features)
#(A,V,L)

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

class ReverseLayerF(Function):
    """
    Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
    """
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

class LogSoftmax_new(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LogSoftmax_new, self).__init__()
        self.dim = dim
        self.eps = eps
    def forward(self, x):
        ex = torch.exp(x)
        ex = ex + (ex==0.)*self.eps
        sumx = torch.sum(ex, axis=self.dim, keepdim=True)
        return torch.log(ex/sumx)

        add_one = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(audio_h).to(text.device)

# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, feature_size=[512,512,512], emotion_cls=4, h_dims=128, dropout=0.25, layers=4):
        super(MISA, self).__init__()
        self.text_size = feature_size[2]
        self.visual_size = feature_size[1]
        self.acoustic_size = feature_size[0]
        self.hidden_size = h_dims
        self.rnncell = 'lstm'
        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.output_size = output_size = emotion_cls
        self.dropout_rate = dropout_rate = dropout
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.input_sizes[2], self.hidden_sizes[2], dropout=dropout_rate)
        self.video_subnet = VisionSubNet(self.input_sizes[1], self.hidden_sizes[1], self.hidden_sizes[1], num_layers=layers, dropout=dropout_rate)
        self.text_subnet = SubNet(self.input_sizes[0], self.hidden_sizes[0], dropout=dropout_rate)

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0], out_features=self.hidden_size))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(self.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1], out_features=self.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(self.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2], out_features=self.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(self.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_a.add_module('private_a_activation_1', nn.Sigmoid())
        
        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))

        ##########################################
        # fusion
        ##########################################
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.hidden_size*6, out_features=self.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.hidden_size*3, out_features= output_size))
        self.fusion.add_module('softmax', nn.LogSoftmax(dim=-1))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        ##########################################
        #loss functions
        ##########################################
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()

    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)


        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def get_cmd_loss(self,):
        # losses between shared states
        loss = self.loss_cmd(self.utt_shared_t, self.utt_shared_v, 5)
        loss += self.loss_cmd(self.utt_shared_t, self.utt_shared_a, 5)
        loss += self.loss_cmd(self.utt_shared_a, self.utt_shared_v, 5)
        loss = loss/3.0
        return loss

    def get_diff_loss(self,):
        shared_t,shared_v,shared_a = self.utt_shared_t,self.utt_shared_v,self.utt_shared_a
        private_t,private_v,private_a = self.utt_private_t,self.utt_private_v,self.utt_private_a
        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)
        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)
        return loss

    def get_recon_loss(self,):
        loss = self.loss_recon(self.utt_t_recon, self.utt_t_orig)
        loss += self.loss_recon(self.utt_v_recon, self.utt_v_orig)
        loss += self.loss_recon(self.utt_a_recon, self.utt_a_orig)
        loss = loss/3.0
        return loss

    def forward(self, inputs, input_padding_masks):
        audio, vision, text = inputs   #shape: (B,L,D)
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l = input_padding_masks  #shape: (B,L), where True indicates paddings
        batch_size = text.size(0)
        # extract features from textual modality
        utterance_text,_ = self.text_subnet(text, key_padding_mask_l)

        # extract features from visual modality
        utterance_video,_ = self.video_subnet(vision, key_padding_mask_v)

        # extract features from acoustic modality
        utterance_audio,_ = self.audio_subnet(audio, key_padding_mask_a)

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)
        
        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        output = self.fusion(h)

        res = {'label':output, 'embed':h}

        return res