"""Original CausalSelfAttention vs MHA + RoPE."""

import math

import torch
import torch.nn as nn

from cohebot.attention.mha import MultiHeadAttention

from . import make_data, print_table, run


class CausalSelfAttention(nn.Module):
    """Original attention from init commit (no RoPE, fused QKV)."""

    def __init__(self, embed_dim, num_heads, max_seq_len=4096, dropout=0.1, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask: torch.Tensor = self.causal_mask  # type: ignore[assignment]
        attn = attn.masked_fill(causal_mask[:T, :T], float("-inf"))
        attn = self.attn_dropout(torch.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.resid_dropout(self.out_proj(out))


def main():
    data = make_data()
    results = {
        "Original (no RoPE)": run(CausalSelfAttention, data),
        "MHA + RoPE": run(MultiHeadAttention, data),
    }
    print_table(results)


if __name__ == "__main__":
    main()
