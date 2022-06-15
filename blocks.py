import numpy as np
import torch 
import torch.nn as nn 

"""
Author  : Nguyễn Thị Hồng Phúc
Tutorial: https://www.tensorflow.org/text/tutorials/transformer
"""

MAX_TOKENS = 128

"""
The look-ahead mask is used to mask the future tokens in a sequence. 
In other words, the mask indicates which entries should not be used.

This means that to predict the third token, only the first and second token will be used. 
Similarly to predict the fourth token, only the first, second and the third tokens will be used and so on.

"""
def create_look_ahead_mask(size):
  mask = 1 - torch.tril(torch.ones(size,size))
  return mask  # (seq_len, seq_len)

def create_padding_mask(inp):
    padding_msk = (inp == 0).float()  # input == 0 -> 1: mask, 1 -> 0: no mask 
    bz, seg_len = padding_msk.shape
    padding_msk = padding_msk.view(bz, 1, 1, seg_len)
    return padding_msk

"""
Positional Encoding
"""
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(torch.arange(position).unsqueeze(1),
                            torch.arange(d_model).unsqueeze(0),
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i + 1
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    
    return pos_encoding

"""""
Scaled Dot-Product Attention
Attention(q, k, v) = softmax(q.(k.T)) / (sqrt(k)) # or sqrt(v), sqrt(v)

Return: batch_size, n_heads, seq_len, dept
"""""
def scaled_dot_product_attn(q, k, v, msk):
    qk = torch.matmul(q, k.transpose(2, 3))
    att = torch.pow(qk, 0.5)
    if msk is not None:
        att += (msk * -1e9)   # torch.softmax(negativa values) ~ 0 -> remove padding when training model
    
    att = torch.softmax(att, dim=-1)
    output = torch.matmul(att, v)

    return output, att

"""
Multi-Head Attention

Return: bz, sq_len, d_model
"""
class MultiHeadAttn(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        assert d_model % n_heads == 0

        self.depth = d_model // n_heads

        self.query = nn.Linear(d_model, n_heads * self.depth, bias=False)
        self.key   = nn.Linear(d_model, n_heads * self.depth, bias=False)
        self.value = nn.Linear(d_model, n_heads * self.depth, bias=False)

        self.fc    = nn.Linear(d_model, n_heads * self.depth, bias=False)

    def split_head(self, x, bz):
        """"
        x_dimension: bz, sequence_len, d_model
        Split the last dimension into (n_heads, depth).
        Reshape the x_dimension into (batch_size, n_heads, seq_len, depth)        

        """
        x = x.view(bz, -1, self.n_heads, self.depth)
        x = x.view(bz, self.n_heads, -1, self.depth)

        return x

    def forward(self, v, k, q, msk):
        bz = v.shape[0]
        
        q = self.query(q)  
        k = self.key(k)
        v = self.value(v)

        """
        Reshape q,k,v into (batch_size, n_heads, seq_len, depth) before directing to scaled_dot_product_attn
        """ 
        q = self.split_head(q, bz)   
        k = self.split_head(k, bz)
        v = self.split_head(v, bz)

        scaled_attn, weight_attn =  scaled_dot_product_attn(q, k, v, msk) 
        scaled_attn = scaled_attn.view(bz, -1, self.n_heads, self.depth)
        concat_attn = scaled_attn.view(bz, -1, self.d_model)

        output = self.fc(concat_attn)
        return output, weight_attn

"""
Position-wise Feed-Forward Networks
"""
def point_wise_feed_forward_network(d_model, dff):
    return nn.Sequential(
                    nn.Linear(d_model, dff),
                    nn.ReLU(),
                    nn.Linear(dff, d_model)
            )

"""
Sublayers
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttn(d_model=d_model, n_heads=n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, msk):
        attn_output, _ = self.mha(x, x, x, msk) 
        attn_output = self.dropout(attn_output)
        output1 = self.layernorm(x + attn_output)

        ffn_output = self.ffn(output1)
        ttn_output = self.dropout(ffn_output)
        output2 = self.layernorm(output1 + ttn_output)

        return output2

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttn(d_model=d_model, n_heads=n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1  = self.mha(x, x, x, look_ahead_mask) 
        attn1 = self.dropout(attn1)
        out1 = self.layernorm(x + attn1)
    
        attn2, attn_weights_block2 = self.mha(v=enc_output, k=enc_output, q=out1, msk=padding_mask)
        attn2 = self.dropout(attn2)
        out2 = self.layernorm(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout(ffn_output)
        out3 = self.layernorm(out2 + ffn_output)

        return out3, attn_weights_block1, attn_weights_block2

"""
Transformer Encoder and Decoder
"""
class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, dff, input_vocab_size, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(MAX_TOKENS, d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model, n_heads=n_heads, dff=dff, dropout=dropout)
            for _ in range(n_layers)]
        
        self.dropout = nn.Dropout()

    def forward(self, x, msk):
        seg_len = x.shape[1]

        # adding embedding and position encoding
        x = self.embedding(x)                   #  (batch_size=64, input_seq_len=62, d_model=512)
        x *= self.d_model ** 0.5
        x += self.pos_encoding[:, :seg_len, :]  #  pos_encoding=(batch_size=64, input_seq_len=62, d_model=512)
        
        x = self.dropout(x)
        for i in range(self.n_layers):
            x = self.enc_layers[i](x, msk)

        return x 


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, dff, target_vocab_size, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(MAX_TOKENS, d_model)

        self.dec_layers = [
            DecoderLayer(d_model=d_model, n_heads=n_heads, dff=dff, dropout=dropout)
            for _ in range(n_layers)]
        
        self.dropout = nn.Dropout()

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seg_len = x.shape[1]
        attn_wts = {}

        # adding embedding and position encoding
        x = self.embedding(x)                  
        x *= self.d_model ** 0.5
        x += self.pos_encoding[:, :seg_len, :] 
        
        x = self.dropout(x)
        for i in range(self.n_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

        attn_wts[f'decoder_layer{i+1}_block1'] = block1
        attn_wts[f'decoder_layer{i+1}_block2'] = block2

        return x, attn_wts




