import torch
from torch import nn
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.mcattention import MCAttention
import math

class MCTransformerEncoder(nn.Module):
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
            new_layer = MCTransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(3)])

    def residual_add(self, x, residual):
        for i in range(len(x)):
            x[i] = residual[i] + F.dropout(x[i], p=self.res_dropout, training=self.training)
        return x

    def forward(self, x, padding_masks):
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
        x = list(x)
        for i in range(len(x)):
            x[i] = self.preprocess(x[i])
        
        # encoder layers
        intermediates = [x]
        attn_intermediates = []
        for layer in self.layers:
            if len(intermediates)>0:
                x_residual = intermediates[-1]
            x, attn_weights = layer(x, padding_masks)
            x = self.residual_add(x,x_residual)
            intermediates.append(x)
            attn_intermediates.append(attn_weights)

        if self.normalize:
            for i in range(len(x)):
                x[i] = self.layer_norms[i](x[i])

        return x, intermediates, attn_intermediates

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

class MCTransformerEncoderLayer(nn.Module):
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

    def __init__(self, embed_dim, num_heads=4, relu_dropout=0.1, res_dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.mcattn = MCAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = False

        self.fc1 = nn.ModuleList([Linear(self.embed_dim, 4*self.embed_dim) for _ in range(3)])   # The "Add & Norm" part in the paper
        self.fc2 = nn.ModuleList([Linear(4*self.embed_dim, self.embed_dim) for _ in range(3)])
        self.layer_norms_0 = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(3)])
        self.layer_norms_1 = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(3)])

    def forward(self, inputs, input_padding_masks):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            input_padding_masks (Tensor): input to the layer of shape `(batch, seq_len)`
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        inputs = list(inputs)
        residual = inputs
        x = self.maybe_layer_norm_0(inputs, before=True)
        x, attn_weights = self.mcattn(x, input_padding_masks)
        x = list(x)
        x = self.residual_add(x, residual)
        x = self.maybe_layer_norm_0(x, after=True)

        residual = x
        x = self.maybe_layer_norm_1(x, before=True)
        x = self.fc(x)
        x = self.residual_add(x, residual)
        x = self.maybe_layer_norm_1(x, after=True)
        return x, attn_weights

    def residual_add(self, x, residual):
        for i in range(len(x)):
            x[i] = residual[i] + F.dropout(x[i], p=self.res_dropout, training=self.training)
        return x

    def maybe_layer_norm_0(self, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            for i in range(len(x)):
                x[i] = self.layer_norms_0[i](x[i])
        return x

    def maybe_layer_norm_1(self, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            for i in range(len(x)):
                x[i] = self.layer_norms_1[i](x[i])
        return x

    def fc(self, x):
        for i in range(len(x)):
            x[i] = self.fc2[i](F.dropout(F.relu(self.fc1[i](x[i])), p=self.relu_dropout, training=self.training))
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
