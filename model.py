import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Function
from collections import OrderedDict
from modules.transformer import TransformerEncoder, TransformerDecoder, MITransformerEncoder, PMREncoder, EMTEncoder
from modules.mctransformer import MCTransformerEncoder

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

class RECLoss(nn.Module):
    def __init__(self):
        super(RECLoss, self).__init__()
        self.eps = torch.FloatTensor([1e-4])
        self.loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
        eps = self.eps.to(pred.device)
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2])
        loss = self.loss(pred*mask, target*mask) / (torch.sum(mask) + eps)
        return loss

class TFRNet(nn.Module):
    def __init__(self, feature_size=[512,512,512], max_len_ls=[400,50,50], emotion_cls=4, h_dims=128, dropout=0.25, layers=4):
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
        self.max_len_ls = max_len_ls

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

        self.trans_l_final = self.get_network(self_type='l_final', layers=2)
        self.trans_a_final = self.get_network(self_type='a_final', layers=2)
        self.trans_v_final = self.get_network(self_type='v_final', layers=2)

        self.generator_l = nn.Linear(self.d_l*3, self.orig_d_l)
        self.generator_a = nn.Linear(self.d_a*3, self.orig_d_a)
        self.generator_v = nn.Linear(self.d_v*3, self.orig_d_v)
        self.gen_loss = RECLoss()

        # Projection layers
        self.fusion_subnet = GATE_F(emotion_cls=emotion_cls, embedding_dim=3*h_dims, h_dims=h_dims, dropout=dropout)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_l, self.attn_dropout, self.num_heads, True
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_a, self.attn_dropout_a, self.num_heads, True
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_v, self.attn_dropout_v, self.num_heads, True
        elif self_type == 'l_final':
            embed_dim, attn_dropout, num_heads, position_embedding = self.max_len_ls[2], self.attn_dropout, 5, False
        elif self_type == 'a_final':
            embed_dim, attn_dropout, num_heads, position_embedding = self.max_len_ls[0], self.attn_dropout, 5, False
        elif self_type == 'v_final':
            embed_dim, attn_dropout, num_heads, position_embedding = self.max_len_ls[1], self.attn_dropout, 5, False
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  layers=layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask, 
                                  position_embedding=position_embedding)

    def forward(self, inputs, inputs_m, input_padding_masks, input_missing_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        key_padding_mask_a, key_padding_mask_v, and key_padding_mask_l should have dimension [batch_size, seq_len]
        """
        audio, vision, text = inputs
        audio_m, vision_m, text_m = inputs_m
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l = input_padding_masks
        missing_mask_a, missing_mask_v, missing_mask_l = input_missing_masks
        x_a, x_v, x_l = audio_m.transpose(1, 2), vision_m.transpose(1, 2), text_m.transpose(1, 2)
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
        h_ls_n = self.trans_l_final(h_ls.permute(2,1,0))[0].permute(1,2,0)  # (L,B,D) -> (D,B,L) -> (B,L,D)
        h_ls = h_ls.transpose(0, 1)   # shape: (L,B,D) -> (B,L,D)

        # (L,V) --> A
        h_a = self.trans_a(proj_x_a, key_padding_mask=key_padding_mask_a)
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l, key_padding_mask_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v, key_padding_mask_v)
        h_as = torch.cat([h_a[0], h_a_with_ls[0], h_a_with_vs[0]], dim=2)
        h_as_n = self.trans_a_final(h_as.permute(2,1,0))[0].permute(1,2,0)
        h_as = h_as.transpose(0, 1)

        # (L,A) --> V
        h_v = self.trans_v(proj_x_v, key_padding_mask=key_padding_mask_v)
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l, key_padding_mask_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a, key_padding_mask_a)
        h_vs = torch.cat([h_v[0], h_v_with_ls[0], h_v_with_as[0]], dim=2)
        h_vs_n = self.trans_v_final(h_vs.permute(2,1,0))[0].permute(1,2,0)
        h_vs = h_vs.transpose(0, 1)

        text_ = self.generator_l(h_ls_n)
        audio_ = self.generator_a(h_as_n)
        vision_ = self.generator_v(h_vs_n)

        text_gen_loss = self.gen_loss(text_, text, missing_mask_l)
        audio_gen_loss = self.gen_loss(audio_, audio, missing_mask_a)
        vision_gen_loss = self.gen_loss(vision_, vision, missing_mask_v)        

        losses = {'text_gen_loss':text_gen_loss,'audio_gen_loss':audio_gen_loss,'vision_gen_loss':vision_gen_loss}
        # fusion&classification
        res = self.fusion_subnet((h_as,h_vs,h_ls),input_padding_masks)
        res.update(losses)
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
                                        position_embedding=True)       
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

    def __init__(self, n_moments=5):
        super(CMD, self).__init__()
        self.n_moments = n_moments

    def forward(self, x1, x2):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(self.n_moments - 1):
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

class JS(nn.Module):
    def __init__(self, get_softmax=True, reduction='batchmean'):
        super(JS, self).__init__()
        self.KL = nn.KLDivLoss(reduction=reduction)
        self.get_softmax = get_softmax

    def forward(self, x1, x2):
        if self.get_softmax:
            x1 = F.softmax(x1,-1)
            x2 = F.softmax(x2,-1)
        log_mean = ((x1+x2)/2).log()
        return (self.KL(log_mean, x1) + self.KL(log_mean, x2))/2

class Cosine(nn.Module):
    def __init__(self, reduction='mean'):
        super(Cosine, self).__init__()
        self.reduction = reduction
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, x1, x2):
        sim = self.cos(x1,x2)
        if self.reduction=='mean':
            sim = torch.mean(sim)
        elif self.reduction=='sum':
            sim = torch.sum(sim)
        return sim

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
        utterance_text = self.text_subnet(text, key_padding_mask_l)

        # extract features from visual modality
        utterance_video = self.video_subnet(vision, key_padding_mask_v)

        # extract features from acoustic modality
        utterance_audio = self.audio_subnet(audio, key_padding_mask_a)

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

######################################## PMR ########################################
#input: (batch_size, seq_len, n_features)
#(A,V,L)
class PMR(nn.Module):
    def __init__(self, feature_size=[512,512,512], emotion_cls=4, h_dims=128, dropout=0.25, layers=4):
        super(PMR, self).__init__()
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
        self.pmr = PMREncoder(embed_dim=h_dims,
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
        (h_as,h_vs,h_ls),_,_ = self.pmr((proj_x_a,proj_x_v,proj_x_l), input_padding_masks)
        h_as,h_vs,h_ls = h_as.transpose(0, 1),h_vs.transpose(0, 1), h_ls.transpose(0, 1)   # shape: (L,B,D) -> (B,L,D)
        # fusion&classification
        res = self.fusion_subnet((h_as,h_vs,h_ls),input_padding_masks)
        return res


######################################## EMT ########################################
#input: (batch_size, seq_len, n_features)
#(A,V,L)
class EMT(nn.Module):
    def __init__(self, feature_size=[512,512,512], emotion_cls=4, h_dims=128, dropout=0.25, layers=4):
        super(EMT, self).__init__()
        self.orig_d_a, self.orig_d_v, self.orig_d_l = feature_size
        self.d = h_dims
        self.num_heads = 4
        self.layers = layers
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = dropout
        self.attn_dropout = 0.0
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d, kernel_size=1, padding=0, bias=False)

        # 2. Multimodal collaborative attentions
        self.mct = EMTEncoder(embed_dim=h_dims,
                                        num_heads=self.num_heads,
                                        layers=layers,
                                        relu_dropout=self.relu_dropout,
                                        res_dropout=self.res_dropout,
                                        embed_dropout=self.embed_dropout,
                                        position_embedding=True)       

        self.generator_l = nn.Linear(self.d, self.orig_d_l)
        self.generator_a = nn.Linear(self.d, self.orig_d_a)
        self.generator_v = nn.Linear(self.d, self.orig_d_v)
        self.gen_loss = RECLoss()

        ## projector
        ## gmc_tokens: global multimodal context
        gmc_tokens_dim = 3 * self.d
        self.gmc_tokens_projector = Projector(gmc_tokens_dim, gmc_tokens_dim)
        self.text_projector = Projector(self.d, self.d)
        self.audio_projector = Projector(self.d, self.d)
        self.video_projector = Projector(self.d, self.d)

        ## predictor
        self.gmc_tokens_predictor = Predictor(gmc_tokens_dim, gmc_tokens_dim//2, gmc_tokens_dim)
        self.text_predictor = Predictor(self.d, self.d//2, self.d)
        self.audio_predictor = Predictor(self.d, self.d//2, self.d)
        self.video_predictor = Predictor(self.d, self.d//2, self.d)

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

    def forward(self, inputs, inputs_m, input_padding_masks, input_missing_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        key_padding_mask_a, key_padding_mask_v, and key_padding_mask_l should have dimension [batch_size, seq_len]
        """
        audio, vision, text = inputs
        #audio_ref, vision_ref, text_ref = audio.clone().detach(), vision.clone().detach(), text.clone().detach()
        audio_m, vision_m, text_m = inputs_m
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l = input_padding_masks
        missing_mask_a, missing_mask_v, missing_mask_l = input_missing_masks
        # Project the textual/visual/audio features
        proj_x_a_m, proj_x_v_m, proj_x_l_m = self.conv_proj(audio_m, vision_m, text_m)
        proj_x_a, proj_x_v, proj_x_l = self.conv_proj(audio, vision, text)

        h_as_m,h_vs_m,h_ls_m,ctx_m = self.mct((proj_x_a_m,proj_x_v_m,proj_x_l_m), input_padding_masks)       
        h_as,h_vs,h_ls,ctx = self.mct((proj_x_a,proj_x_v,proj_x_l), input_padding_masks)    #h_as: (L,B,D), ctx: (3,B,D)
        
        # low-level feature reconstruction
        audio_ = h_as_m if self.orig_d_a == self.d else self.generator_a(h_as_m.permute(1, 0, 2))       # (L,B,D) -> (B,L,D)
        vision_ = h_vs_m if self.orig_d_v == self.d else self.generator_v(h_vs_m.permute(1, 0, 2)) 
        text_ = h_ls_m if self.orig_d_l == self.d else self.generator_l(h_ls_m.permute(1, 0, 2))
        audio_gen_loss = self.gen_loss(audio_, audio, missing_mask_a)
        vision_gen_loss = self.gen_loss(vision_, vision, missing_mask_v)   
        text_gen_loss = self.gen_loss(text_, text, missing_mask_l)

        losses = {'text_gen_loss':text_gen_loss,'audio_gen_loss':audio_gen_loss,'vision_gen_loss':vision_gen_loss}

        ctx_utt = ctx.permute(1,0,2).reshape(-1,3*self.d)
        ctx_utt_m = ctx_m.permute(1,0,2).reshape(-1,3*self.d)
        text_utt, audio_utt, video_utt = h_ls[0], h_as[0], h_vs[0]
        text_utt_m, audio_utt_m, video_utt_m = h_ls_m[0], h_as_m[0], h_vs_m[0]

        # high-level feature attraction via SimSiam
        ## projector
        z_gmc_tokens = self.gmc_tokens_projector(ctx_utt)
        z_text = self.text_projector(text_utt)
        z_audio = self.audio_projector(audio_utt)
        z_video = self.video_projector(video_utt)

        z_gmc_tokens_m = self.gmc_tokens_projector(ctx_utt_m)
        z_text_m = self.text_projector(text_utt_m)
        z_audio_m = self.audio_projector(audio_utt_m)
        z_video_m = self.video_projector(video_utt_m)

        ## predictor
        p_gmc_tokens = self.gmc_tokens_predictor(z_gmc_tokens)
        p_text = self.text_predictor(z_text)
        p_audio = self.audio_predictor(z_audio)
        p_video = self.video_predictor(z_video)

        p_gmc_tokens_m = self.gmc_tokens_predictor(z_gmc_tokens_m)
        p_text_m = self.text_predictor(z_text_m)
        p_audio_m = self.audio_predictor(z_audio_m)
        p_video_m = self.video_predictor(z_video_m)

        intermedias = {
            'z_gmc_tokens': z_gmc_tokens.detach(),
            'p_gmc_tokens': p_gmc_tokens,
            'z_text': z_text.detach(),
            'p_text': p_text,
            'z_audio': z_audio.detach(),
            'p_audio': p_audio,
            'z_video': z_video.detach(),
            'p_video': p_video,
            'z_gmc_tokens_m': z_gmc_tokens_m.detach(),
            'p_gmc_tokens_m': p_gmc_tokens_m,
            'z_text_m': z_text_m.detach(),
            'p_text_m': p_text_m,
            'z_audio_m': z_audio_m.detach(),
            'p_audio_m': p_audio_m,
            'z_video_m': z_video_m.detach(),
            'p_video_m': p_video_m            
        }

        # Global Feature Alignment
        h_as_m,h_vs_m,h_ls_m = h_as_m.transpose(0, 1),h_vs_m.transpose(0, 1), h_ls_m.transpose(0, 1)   # shape: (L,B,D) -> (B,L,D) 
        h_as,h_vs,h_ls = h_as.transpose(0, 1),h_vs.transpose(0, 1), h_ls.transpose(0, 1) 
        
        # fusion & classification
        res = self.fusion_subnet((h_as,h_vs,h_ls),input_padding_masks)
        res_m = self.fusion_subnet((h_as_m,h_vs_m,h_ls_m),input_padding_masks)
        res_final = {'embed':res['embed'], 'embed_m':res_m['embed'], 
                    'label':res['label'], 'label_m':res_m['label']}
        res_final.update(losses)
        res_final.update(intermedias)
        return res_final

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_dim, output_dim),
                                 nn.BatchNorm1d(output_dim, affine=False))

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, input_dim, pred_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, pred_dim, bias=False),
                                 nn.BatchNorm1d(pred_dim),
                                 nn.ReLU(inplace=True),  # hidden layer
                                 nn.Linear(pred_dim, output_dim))  # output layer

    def forward(self, x):
        return self.net(x)



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
        self.attn_dropout = 0.0
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
        
        self.trans_l_final = self.get_network(self_type='l_final', layers=2)
        self.trans_a_final = self.get_network(self_type='a_final', layers=2)
        self.trans_v_final = self.get_network(self_type='v_final', layers=2)

        self.generator_l = nn.Conv1d(self.d, self.orig_d_l, kernel_size=1, padding=0, bias=False)
        self.generator_a = nn.Conv1d(self.d, self.orig_d_a, kernel_size=1, padding=0, bias=False)
        self.generator_v = nn.Conv1d(self.d, self.orig_d_v, kernel_size=1, padding=0, bias=False)
        self.gen_loss = RECLoss()          

        # Projection layers
        self.fusion_subnet = GATE_F(emotion_cls=emotion_cls, embedding_dim=h_dims, h_dims=h_dims//2, dropout=dropout)
        self.predictor = nn.Sequential(nn.Linear(h_dims*3//2, h_dims*3//4, bias=False),
                                        nn.BatchNorm1d(h_dims*3//4),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(h_dims*3//4, h_dims*3//2)) # output layer

    def conv_proj(self, audio, vision, text):
        x_a, x_v, x_l = audio.transpose(1, 2), vision.transpose(1, 2), text.transpose(1, 2)
        proj_x_a = x_a if self.orig_d_a == self.d else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d else self.proj_v(x_v)
        proj_x_l = x_l if self.orig_d_l == self.d else self.proj_l(x_l)        
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        return proj_x_a, proj_x_v, proj_x_l

    def conv_proj_decode(self, audio, vision, text):
        # input: (L,B,D) -> (B,D,L)
        x_a, x_v, x_l = audio.permute(1, 2, 0), vision.permute(1, 2, 0), text.permute(1, 2, 0)
        proj_x_a = x_a if self.orig_d_a == self.d else self.generator_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d else self.generator_v(x_v)
        proj_x_l = x_l if self.orig_d_l == self.d else self.generator_l(x_l)        
        proj_x_a = proj_x_a.permute(0, 2, 1)   # (B,D,L) -> (B,L,D)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        proj_x_l = proj_x_l.permute(0, 2, 1)
        return proj_x_a, proj_x_v, proj_x_l

    def get_network(self, self_type='a_final', layers=-1):
        if self_type == 'a_final':
            embed_dim, attn_dropout, num_heads, position_embedding = self.d, self.attn_dropout, 4, False
        elif self_type == 'v_final':
            embed_dim, attn_dropout, num_heads, position_embedding = self.d, self.attn_dropout, 4, False
        elif self_type == 'l_final':
            embed_dim, attn_dropout, num_heads, position_embedding = self.d, self.attn_dropout, 4, False
        else:
            raise ValueError("Unknown network type")
        
        return TransformerDecoder(embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  layers=layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=False, 
                                  position_embedding=position_embedding)

    def forward(self, inputs, inputs_m, input_padding_masks, input_missing_masks):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        key_padding_mask_a, key_padding_mask_v, and key_padding_mask_l should have dimension [batch_size, seq_len]
        """
        audio, vision, text = inputs
        audio_m, vision_m, text_m = inputs_m
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l = input_padding_masks
        missing_mask_a, missing_mask_v, missing_mask_l = input_missing_masks
        # Project the textual/visual/audio features
        proj_x_a_m, proj_x_v_m, proj_x_l_m = self.conv_proj(audio_m, vision_m, text_m)
        proj_x_a, proj_x_v, proj_x_l = self.conv_proj(audio, vision, text)
        
        (h_as_m,h_vs_m,h_ls_m),_,attn_m = self.mct((proj_x_a_m,proj_x_v_m,proj_x_l_m), input_padding_masks)       
        (h_as,h_vs,h_ls),_,attn = self.mct((proj_x_a,proj_x_v,proj_x_l), input_padding_masks)

        # Local Feature Imagination
        h_as_n = self.trans_a_final(proj_x_a_m, h_as_m, h_as_m, key_padding_mask_a)[0]             # (L,B,D)
        h_vs_n = self.trans_v_final(proj_x_v_m, h_vs_m, h_vs_m, key_padding_mask_v)[0]
        h_ls_n = self.trans_l_final(proj_x_l_m, h_ls_m, h_ls_m, key_padding_mask_l)[0]

        audio_, vision_, text_ = self.conv_proj_decode(h_as_n, h_vs_n, h_ls_n)
        audio_gen_loss = self.gen_loss(audio_, audio, missing_mask_a)
        vision_gen_loss = self.gen_loss(vision_, vision, missing_mask_v)   
        text_gen_loss = self.gen_loss(text_, text, missing_mask_l)
        losses = {'text_gen_loss':text_gen_loss,'audio_gen_loss':audio_gen_loss,'vision_gen_loss':vision_gen_loss}

        # Global Feature Alignment
        h_as_m,h_vs_m,h_ls_m = h_as_m.transpose(0, 1),h_vs_m.transpose(0, 1), h_ls_m.transpose(0, 1)   # shape: (L,B,D) -> (B,L,D) 
        h_as,h_vs,h_ls = h_as.transpose(0, 1),h_vs.transpose(0, 1), h_ls.transpose(0, 1) 
        # fusion&classification
        res = self.fusion_subnet((h_as,h_vs,h_ls),input_padding_masks)
        res_m = self.fusion_subnet((h_as_m,h_vs_m,h_ls_m),input_padding_masks)
        pred = self.predictor(res['embed'])
        pred_m = self.predictor(res_m['embed'])               

        res_final = {'embed':res['embed'], 'embed_m':res_m['embed'], 
                    'label':res['label'], 'label_m':res_m['label'], 
                    'pred':pred, 'pred_m':pred_m,
                    'attn':attn[-1], 'attn_m':attn_m[-1]}
        res_final.update(losses)
        return res_final
        
