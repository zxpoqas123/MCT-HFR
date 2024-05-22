import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys

# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
        
        attn_weights = torch.bmm(q, k.transpose(1, 2)) #shape: [bsz*head, tgt_len, src_len]
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]


        # here key_padding_mask is comprised of True and False, where True indicate the position of padding
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len)
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                new_attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
                new_attn_mask.masked_fill_(key_padding_mask, float("-inf"))
                attn_mask = new_attn_mask
            else:
                attn_mask.masked_fill_(key_padding_mask, float("-inf"))

        if attn_mask is not None:
            if len(attn_mask.shape)==3:
                attn_weights += attn_mask
            elif len(attn_mask.shape)==2:
                attn_weights += attn_mask.unsqueeze(0)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, inputs, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(inputs, weight, bias)

