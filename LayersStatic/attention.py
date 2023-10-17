import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask


class StaticAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, dropout=0.1, atts=False, n_size=(96,96), n_heads=8):
        super(StaticAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = atts
        self.dropout = nn.Dropout(dropout)
        self.scores = torch.nn.Parameter(torch.rand(size=(1, n_heads, n_size[0], n_size[1])), requires_grad=True)

    def forward(self, values, attn_mask, B, L, E):
        scale = self.scale or 1. / sqrt(E)
        scores = self.scores.repeat(B, 1, 1, 1)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=values.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        self.d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.value_projection = nn.Linear(d_model, self.d_values * n_heads)
        self.out_projection = nn.Linear(self.d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = values.shape
        H = self.n_heads

        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(values, attn_mask, B, L, self.d_values)
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn