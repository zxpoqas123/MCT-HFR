import torch
from torch import nn
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.multihead_attention import MultiheadAttention
import math


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, position_embedding=True):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        #self.embed_scale = 1
        if position_embedding:
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        else: 
            self.embed_positions = None
        
        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k = None, x_in_v = None, key_padding_mask=None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions    
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # encoder layers
        intermediates = [x]
        attn_intermediates = []
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x, attn_weights = layer(x=x, x_k=x_k, x_v=x_v, key_padding_mask=key_padding_mask)
            else:
                x, attn_weights = layer(x=x, key_padding_mask=key_padding_mask)
            intermediates.append(x)
            attn_intermediates.append(attn_weights)

        if self.normalize:
            x = self.layer_norm(x)

        return x, intermediates, attn_intermediates

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

class TransformerDecoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, position_embedding=True):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        #self.embed_scale = 1
        if position_embedding:
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        else: 
            self.embed_positions = None
        
        self.layers = layers
        self.layers_self_attn = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=False)
            self.layers_self_attn.append(new_layer)

        self.layers_cross_attn = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=False)
            self.layers_cross_attn.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k, x_in_v, key_padding_mask=None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions    
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # encoder layers
        intermediates = [x]
        attn_intermediates = []
        for layer in range(self.layers):
            x, _ = self.layers_self_attn[layer](x=x, key_padding_mask=key_padding_mask)
            x, attn_weights = self.layers_cross_attn[layer](x=x, x_k=x_k, x_v=x_v, key_padding_mask=key_padding_mask)
            intermediates.append(x)
            attn_intermediates.append(attn_weights)

        if self.normalize:
            x = self.layer_norm(x)

        return x, intermediates, attn_intermediates

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class MITransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, position_embedding=True, beta_shift=0.5):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        #self.embed_scale = 1
        if position_embedding:
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        else: 
            self.embed_positions = None
        
        self.attn_mask = attn_mask
        self.eps = 1e-6
        self.beta_shift = beta_shift
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k = None, x_in_v = None, key_padding_mask=None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions    
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # encoder layers
        intermediates = [x]
        attn_intermediates = []
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                H, attn_weights = layer(x=x, x_k=x_k, x_v=x_v, key_padding_mask=key_padding_mask)
                em_norm = x.norm(2, dim=-1)
                H_norm = H.norm(2, dim=-1)
                thresh_hold = (em_norm / (H_norm + self.eps)) * self.beta_shift
                ones = torch.ones(thresh_hold.shape, requires_grad=True).to(x.device)
                gama = torch.min(thresh_hold, ones).unsqueeze(dim=-1)
                x = x + gama * H
            else:
                x, attn_weights = layer(x=x, key_padding_mask=key_padding_mask)
            intermediates.append(x)
            attn_intermediates.append(attn_weights)

        if self.normalize:
            x = self.layer_norm(x)

        return x, intermediates, attn_intermediates

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = False

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None, key_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, attn_weights = self.self_attn(query=x, key=x, value=x, key_padding_mask=key_padding_mask, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True) 
            x, attn_weights = self.self_attn(query=x, key=x_k, value=x_v, key_padding_mask=key_padding_mask, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x, attn_weights

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

class MRU(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.cross_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.proj_weight_sa = Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
        self.proj_weight_ca = Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
        self.proj_bias = Parameter(torch.Tensor(self.embed_dim))
        self.sigmoid = nn.Sigmoid()

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x_s, x_t, key_padding_mask_s=None, key_padding_mask_t=None):
        sa, _ = self.self_attn(query=x_t, key=x_t, value=x_t, key_padding_mask=key_padding_mask_t)
        ca, _ = self.cross_attn(query=x_t, key=x_s, value=x_s, key_padding_mask=key_padding_mask_s)
        G = self.sigmoid(torch.matmul(sa,self.proj_weight_sa)+torch.matmul(ca,self.proj_weight_ca)+self.proj_bias)
        x = G*ca + (1-G)*sa
        x = self.layer_norms[0](x)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.layer_norms[1](x)
        return x

class PMRlayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mru_c2a = MRU(embed_dim,num_heads=num_heads,relu_dropout=relu_dropout,res_dropout=res_dropout)
        self.mru_c2v = MRU(embed_dim,num_heads=num_heads,relu_dropout=relu_dropout,res_dropout=res_dropout)
        self.mru_c2l = MRU(embed_dim,num_heads=num_heads,relu_dropout=relu_dropout,res_dropout=res_dropout)
        self.mru_a2c = MRU(embed_dim,num_heads=num_heads,relu_dropout=relu_dropout,res_dropout=res_dropout)
        self.mru_v2c = MRU(embed_dim,num_heads=num_heads,relu_dropout=relu_dropout,res_dropout=res_dropout)
        self.mru_l2c = MRU(embed_dim,num_heads=num_heads,relu_dropout=relu_dropout,res_dropout=res_dropout)

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.ctx_len = 3
        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = LayerNorm(self.embed_dim)
        self.proj_weight_zc1 = Parameter(torch.Tensor(self.ctx_len*self.embed_dim, self.ctx_len*self.embed_dim))
        self.proj_weight_zc2 = Parameter(torch.Tensor(self.ctx_len*self.embed_dim, self.ctx_len*self.embed_dim))
        self.proj_weight_zc3 = Parameter(torch.Tensor(self.ctx_len*self.embed_dim, self.ctx_len*self.embed_dim))
        self.U = Parameter(torch.Tensor(self.ctx_len*self.embed_dim,1))
        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = torch.nn.Tanh()

    def cal_coeff(self, x, w):
        # x: L,B,D
        bsz = x.shape[1]
        x = x.permute(1,0,2).reshape(bsz,-1) # x: B,L*D
        x = torch.matmul(self.tanh(torch.matmul(x,w)),self.U) # B,1
        return x

    def forward(self, inputs, input_padding_masks):        
        audio, vision, text, ctx = inputs  # (L,B,D)
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l , ctx_padding_mask = input_padding_masks # (B,L)

        za = self.mru_c2a(ctx, audio, ctx_padding_mask, key_padding_mask_a)
        zv = self.mru_c2v(ctx, vision, ctx_padding_mask, key_padding_mask_v)
        zl = self.mru_c2l(ctx, text, ctx_padding_mask, key_padding_mask_l)

        zc1 = self.mru_a2c(za, ctx, key_padding_mask_a, ctx_padding_mask)
        zc2 = self.mru_v2c(zv, ctx, key_padding_mask_v, ctx_padding_mask)
        zc3 = self.mru_l2c(zl, ctx, key_padding_mask_l, ctx_padding_mask)

        #coeffi = self.softmax(torch.cat([self.cal_coeff(zc1,self.proj_weight_zc1),self.cal_coeff(zc2,self.proj_weight_zc2),self.cal_coeff(zc3,self.proj_weight_zc3)],dim=-1))
        #x = (coeffi[:,0].reshape(1,-1,1))*zc1 + (coeffi[:,1].reshape(1,-1,1))*zc2 + (coeffi[:,2].reshape(1,-1,1))*zc3 
        zc = zc1+zc2+zc3
        '''
        residual = x
        x = self.layer_norms(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        zc = residual + x
        '''
        
        return [za,zv,zl,zc]

class PMREncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
    """

    def __init__(self, embed_dim, num_heads, layers, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, position_embedding=True):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        #self.embed_scale = 1
        self.res_dropout = res_dropout
        if position_embedding:
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        else: 
            self.embed_positions = None

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = PMRlayer(embed_dim,
                                num_heads=num_heads,
                                relu_dropout=relu_dropout,
                                res_dropout=res_dropout)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = False
        if self.normalize:
            self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(3)])

    def masked_avg_pooling(self, x, x_mask_padding):
        # x: Dimension (seq_len, batch_size, n_features)
        # x_mask_padding: Dimension (batch_size, seq_len)
        x_mask_values = torch.logical_not(x_mask_padding).transpose(0,1).unsqueeze(-1)  # True indicates the position of values, Dimension (seq_len,batch_size,1)
        x_avg_pool = ((x*x_mask_values).sum(0))/(x_mask_values.sum(0))  # x_avg_pool: (batch_size, n_features)
        return x_avg_pool

    def forward(self, inputs, input_padding_masks):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        #ctx = torch.cat(inputs, dim=0)
        #ctx_padding_mask = torch.cat(input_padding_masks, dim=1)
        inputs = list(inputs)
        for i in range(len(inputs)):
            inputs[i] = self.preprocess(inputs[i]) 
        audio, vision, text = inputs  # (L,B,D)
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l = input_padding_masks # (B,L)
        ctx = torch.cat(inputs, dim=0)
        ctx_padding_mask = torch.cat(input_padding_masks, dim=1)        
        #ctx = torch.stack((self.masked_avg_pooling(audio,key_padding_mask_a),self.masked_avg_pooling(vision,key_padding_mask_v),self.masked_avg_pooling(text,key_padding_mask_l))) 
        #ctx_padding_mask = None        
        # encoder layers
        x = [audio, vision, text, ctx]
        padding_masks = [key_padding_mask_a, key_padding_mask_v, key_padding_mask_l, ctx_padding_mask]

        intermediates = [x]
        for layer in self.layers:
            x = layer(x, padding_masks)
            intermediates.append(x)

        if self.normalize:
            for i in range(3):
                x[i] = self.layer_norms[i](x[i])

        return x[0],x[1],x[2]

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def preprocess(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
        
class MPU(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.cross_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.proj_weight_sa = Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
        self.proj_weight_ca = Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
        self.proj_bias = Parameter(torch.Tensor(self.embed_dim))
        self.sigmoid = nn.Sigmoid()

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x_s, x_t, key_padding_mask_s=None, key_padding_mask_t=None):
        ca, _ = self.cross_attn(query=x_t, key=x_s, value=x_s, key_padding_mask=key_padding_mask_s)
        sa, _ = self.self_attn(query=ca, key=ca, value=ca, key_padding_mask=key_padding_mask_t)
        #G = self.sigmoid(torch.matmul(sa,self.proj_weight_sa)+torch.matmul(ca,self.proj_weight_ca)+self.proj_bias)
        #x = G*ca + (1-G)*sa
        x = self.layer_norms[0](sa)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.layer_norms[1](x)
        return x

class EMTlayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mru_c2a = MPU(embed_dim,num_heads=num_heads,relu_dropout=relu_dropout,res_dropout=res_dropout)
        self.mru_c2v = MPU(embed_dim,num_heads=num_heads,relu_dropout=relu_dropout,res_dropout=res_dropout)
        self.mru_c2l = MPU(embed_dim,num_heads=num_heads,relu_dropout=relu_dropout,res_dropout=res_dropout)
        self.mru_a2c = MPU(embed_dim,num_heads=num_heads,relu_dropout=relu_dropout,res_dropout=res_dropout)
        self.mru_v2c = MPU(embed_dim,num_heads=num_heads,relu_dropout=relu_dropout,res_dropout=res_dropout)
        self.mru_l2c = MPU(embed_dim,num_heads=num_heads,relu_dropout=relu_dropout,res_dropout=res_dropout)

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.ctx_len = 3
        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = LayerNorm(self.embed_dim)
        self.proj_weight_zc1 = Parameter(torch.Tensor(self.ctx_len*self.embed_dim, self.ctx_len*self.embed_dim))
        self.proj_weight_zc2 = Parameter(torch.Tensor(self.ctx_len*self.embed_dim, self.ctx_len*self.embed_dim))
        self.proj_weight_zc3 = Parameter(torch.Tensor(self.ctx_len*self.embed_dim, self.ctx_len*self.embed_dim))
        self.U = Parameter(torch.Tensor(self.ctx_len*self.embed_dim,1))
        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = torch.nn.Tanh()

    def cal_coeff(self, x, w):
        # x: L,B,D
        bsz = x.shape[1]
        x = x.permute(1,0,2).reshape(bsz,-1) # x: B,L*D
        x = torch.matmul(self.tanh(torch.matmul(x,w)),self.U) # B,1
        return x

    def forward(self, inputs, input_padding_masks):        
        audio, vision, text, ctx = inputs  # (L,B,D)
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l , ctx_padding_mask = input_padding_masks # (B,L)

        za = self.mru_c2a(ctx, audio, ctx_padding_mask, key_padding_mask_a)
        zv = self.mru_c2v(ctx, vision, ctx_padding_mask, key_padding_mask_v)
        zl = self.mru_c2l(ctx, text, ctx_padding_mask, key_padding_mask_l)

        zc1 = self.mru_a2c(za, ctx, key_padding_mask_a, ctx_padding_mask)
        zc2 = self.mru_v2c(zv, ctx, key_padding_mask_v, ctx_padding_mask)
        zc3 = self.mru_l2c(zl, ctx, key_padding_mask_l, ctx_padding_mask)

        #coeffi = self.softmax(torch.cat([self.cal_coeff(zc1,self.proj_weight_zc1),self.cal_coeff(zc2,self.proj_weight_zc2),self.cal_coeff(zc3,self.proj_weight_zc3)],dim=-1))
        #x = (coeffi[:,0].reshape(1,-1,1))*zc1 + (coeffi[:,1].reshape(1,-1,1))*zc2 + (coeffi[:,2].reshape(1,-1,1))*zc3
        

        #residual = x
        #x = self.layer_norms(x)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=self.relu_dropout, training=self.training)
        #x = self.fc2(x)
        #x = F.dropout(x, p=self.res_dropout, training=self.training)
        #zc = residual + x
        
        zc = zc1+zc2+zc3

        return [za,zv,zl,zc]
        
class EMTEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
    """

    def __init__(self, embed_dim, num_heads, layers, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, position_embedding=True):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        #self.embed_scale = 1
        self.res_dropout = res_dropout
        if position_embedding:
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        else: 
            self.embed_positions = None

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = EMTlayer(embed_dim,
                                num_heads=num_heads,
                                relu_dropout=relu_dropout,
                                res_dropout=res_dropout)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = False
        if self.normalize:
            self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(3)])

    def masked_avg_pooling(self, x, x_mask_padding):
        # x: Dimension (seq_len, batch_size, n_features)
        # x_mask_padding: Dimension (batch_size, seq_len)
        x_mask_values = torch.logical_not(x_mask_padding).transpose(0,1).unsqueeze(-1)  # True indicates the position of values, Dimension (seq_len,batch_size,1)
        x_avg_pool = ((x*x_mask_values).sum(0))/(x_mask_values.sum(0))  # x_avg_pool: (batch_size, n_features)
        return x_avg_pool

    def forward(self, inputs, input_padding_masks):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        inputs = list(inputs)
        for i in range(len(inputs)):
            inputs[i] = self.preprocess(inputs[i]) 
        audio, vision, text = inputs  # (L,B,D)
        key_padding_mask_a, key_padding_mask_v, key_padding_mask_l = input_padding_masks # (B,L)
        ctx = torch.stack((self.masked_avg_pooling(audio,key_padding_mask_a),self.masked_avg_pooling(vision,key_padding_mask_v),self.masked_avg_pooling(text,key_padding_mask_l))) 
        ctx_padding_mask = None        
        #ctx = torch.cat(inputs, dim=0)
        #ctx_padding_mask = torch.cat(input_padding_masks, dim=1)        
        # encoder layers
        x = [audio, vision, text, ctx]
        padding_masks = [key_padding_mask_a, key_padding_mask_v, key_padding_mask_l, ctx_padding_mask]

        intermediates = [x]
        for layer in self.layers:
            x = layer(x, padding_masks)
            intermediates.append(x)

        if self.normalize:
            for i in range(3):
                x[i] = self.layer_norms[i](x[i])

        return x[0],x[1],x[2],x[3]

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def preprocess(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


if __name__ == '__main__':
    encoder = TransformerEncoder(300, 4, 2)
    x = torch.tensor(torch.rand(20, 2, 300))
    print(encoder(x).shape)
