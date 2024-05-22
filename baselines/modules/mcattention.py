import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys

# Code adapted from the fairseq repo.

class MCAttention(nn.Module):
    """Multimodal collaborative attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout_a = 0.1
        self.attn_dropout_v = 0.0
        self.attn_dropout_l = 0.1
        bias = True
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight_a = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_weight_v = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_weight_l = Parameter(torch.Tensor(3 * embed_dim, embed_dim))

        self.in_proj_bias_a = Parameter(torch.Tensor(3 * embed_dim))
        self.in_proj_bias_v = Parameter(torch.Tensor(3 * embed_dim))
        self.in_proj_bias_l = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj_a = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj_l = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight_a)
        nn.init.xavier_uniform_(self.in_proj_weight_v)
        nn.init.xavier_uniform_(self.in_proj_weight_l)
        nn.init.xavier_uniform_(self.out_proj_a.weight)
        nn.init.xavier_uniform_(self.out_proj_v.weight)
        nn.init.xavier_uniform_(self.out_proj_l.weight)

        nn.init.constant_(self.in_proj_bias_a, 0.)
        nn.init.constant_(self.in_proj_bias_v, 0.)
        nn.init.constant_(self.in_proj_bias_l, 0.)
        nn.init.constant_(self.out_proj_a.bias, 0.)
        nn.init.constant_(self.out_proj_v.bias, 0.)
        nn.init.constant_(self.out_proj_l.bias, 0.)

    def in_proj_qkv(self, inputs, weight, bias, bsz):
        Q, K ,V = F.linear(inputs, weight, bias).chunk(3, dim=-1)
        Q = Q * self.scaling
        Q = Q.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  #shape: [bsz*head, len, head_dim]
        K = K.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  #shape: [bsz*head, len, head_dim]
        V = V.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  #shape: [bsz*head, len, head_dim]
        return Q,K,V

    '''
    def generate_len_scale_mask(self, padding_masks):
        # input shape: (bsz, len)
        # return: (bsz, len)
        scale_mask_ls = []
        for padding_mask in padding_masks:
            bsz, max_len = padding_mask.size()
            len_scaling = torch.logical_not(padding_mask).sum(1)**-0.5
            scale_mask = len_scaling.view(bsz, 1).expand(-1,max_len)
            scale_mask_ls.append(scale_mask)
        return torch.cat(scale_mask_ls,dim=1)
    '''

    def generate_len_scale_mask(self, padding_masks):
        # input shape: [(bsz, len1), (bsz, len2), (bsz, len3)]
        # return: (bsz, len1+len2+len3)
        scale_mask_ls = []
        #len_scaling_sum = torch.zeros(bsz).to(padding_masks[0].device)
        for padding_mask in padding_masks:
            bsz, max_len = padding_mask.size()
            len_scaling = torch.logical_not(padding_mask).sum(1)**-0.5 #(bsz,)
            #len_scaling_sum += len_scaling
            scale_mask = len_scaling.view(bsz, 1).expand(-1,max_len)
            scale_mask_ls.append(scale_mask)
        return torch.cat(scale_mask_ls,dim=1)

    def mhca(self, q, key, value, key_padding_mask, scale_mask, bsz, attn_dropout):
        tgt_len = q.size(1)
        src_len = key.size(1)
        attn_weights = torch.bmm(q, key.transpose(1, 2))   #shape: [bsz*head, tgt_len, src_len]
        # here key_padding_mask is comprised of True and False, where True indicate the position of padding
        assert key_padding_mask.shape == (bsz, src_len)
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
        attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
        attn_mask.masked_fill_(key_padding_mask, float("-inf"))
        scale_mask = scale_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
        
        attn_weights = attn_weights*scale_mask+attn_mask
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=attn_dropout, training=self.training)
        attn = torch.bmm(attn_weights, value)     #shape: [bsz*head, tgt_len, head_dims]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn_map = attn_weights.view(bsz, self.num_heads, tgt_len, -1)
        attn_map = attn_map.sum(dim=1) / self.num_heads
        return attn, attn_map

    def forward(self, inputs, input_padding_masks):
        #Input shape: Time x Batch x Channel
        audio, vision, text = inputs
        bsz = audio.size(1)

        Q_a, K_a, V_a = self.in_proj_qkv(audio, self.in_proj_weight_a, self.in_proj_bias_a, bsz)
        Q_v, K_v, V_v = self.in_proj_qkv(vision, self.in_proj_weight_v, self.in_proj_bias_v, bsz)
        Q_l, K_l, V_l = self.in_proj_qkv(text, self.in_proj_weight_l, self.in_proj_bias_l, bsz)

        key_all = torch.cat([K_a,K_v,K_l],dim=1)
        value_all = torch.cat([V_a,V_v,V_l],dim=1)
        key_padding_mask_all = torch.cat(input_padding_masks,dim=1)
        scale_mask_all = self.generate_len_scale_mask(input_padding_masks)

        # (A,V,L)->A
        attn_a, attn_map_a = self.mhca(Q_a, key_all, value_all, key_padding_mask_all, scale_mask_all, bsz, self.attn_dropout_a)
        attn_a = self.out_proj_a(attn_a)
        # (A,V,L)->V
        attn_v, attn_map_v = self.mhca(Q_v, key_all, value_all, key_padding_mask_all, scale_mask_all, bsz, self.attn_dropout_v)
        attn_v = self.out_proj_v(attn_v)
        # (A,V,L)->L
        attn_l, attn_map_l = self.mhca(Q_l, key_all, value_all, key_padding_mask_all, scale_mask_all, bsz, self.attn_dropout_l)
        attn_l = self.out_proj_l(attn_l)
        return (attn_a, attn_v, attn_l), (attn_map_a, attn_map_v, attn_map_l)
