import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import math
import os


# position encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]  # add position encoding
        return self.dropout(x)


# position encoding
def get_sinusoid_encoding_table(n_position, d_model):
    """
    n_position: int the number of position
    d_model: int the number of dimension
    """

    # PE(pos,2i) = sin(pos/10000^(2i/d_model))
    # PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


# padding mask
def get_attn_pad_mask(seq_q, seq_k):
    """
    For masking out the padding part of key sequence.
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


# subsequent mask
def get_attn_subsequence_mask(seq):
    """
    Used in the decoder to mask the future info.
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    device = seq.device
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte().to(device)
    return subsequence_mask


# scaled dot product attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()

        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        # attn = softmax(Q * K^T / sqrt(d_k))
        # context = softmax(attn * V)
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)  # attention : [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V)  # context : [batch_size, n_heads, len_q, d_v]
        return context, attn


# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k). \
            transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k). \
            transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v). \
            transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1). \
            repeat(1, self.n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.d_k).forward(Q, K, V, attn_mask)
        context = context.transpose(1, 2). \
            reshape(batch_size, -1, self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return self.layer_norm(output + residual), attn


# position-wise feed forward net
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)  # [batch_size, seq_len, d_model]


if __name__ == "__main__":
    # positional encoding
    d_model = 10
    pos_encoding = PositionalEncoding(d_model)
    print(pos_encoding.forward(torch.zeros(1, 100, d_model)).shape)

    # padding mask
    seq = torch.tensor([[1, 2, 3, 0, 0],
                        [1, 2, 0, 0, 0]])
    print(get_attn_pad_mask(seq, seq).shape)

    # subsequence mask
    print(get_attn_subsequence_mask(seq).shape)

    # scaled dot product attention
    Q = torch.rand(2, 3, 5)
    K = torch.rand(2, 3, 5)
    V = torch.rand(2, 3, 5)
    attn_mask = torch.zeros(2, 3, 3)
    context, attn = ScaledDotProductAttention(d_k=5).forward(Q, K, V, attn_mask)
    print(context.shape, attn.shape)

    # multi-head attention
    d_model = 10
    d_k = 5
    d_v = 5
    n_heads = 2
    input_Q = torch.rand(2, 3, d_model)
    input_K = torch.rand(2, 3, d_model)
    input_V = torch.rand(2, 3, d_model)
    attn_mask = torch.zeros(2, 3, 3)
    output, attn = MultiHeadAttention(d_model, d_k, d_v, n_heads).forward(input_Q, input_K, input_V, attn_mask)
    print(output.shape, attn.shape)

    # position-wise feed forward net
    d_model = 10
    d_ff = 20
    inputs = torch.rand(2, 3, d_model)
    output = PoswiseFeedForwardNet(d_model, d_ff).forward(inputs)
    print(output.shape)
