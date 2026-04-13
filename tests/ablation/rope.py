"""FA + RoPE vs FA (no positional encoding)."""

import torch.nn as nn
import torch.nn.functional as F

from cohebot.attention.flash import FlashAttention

from . import make_data, print_table, run


class FlashAttentionNoRoPE(nn.Module):
    """FlashAttention with RoPE removed."""

    def __init__(self, embed_dim, num_heads, max_seq_len=4096, dropout=0.1, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        dp = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.resid_dropout(self.out_proj(out))


def main():
    data = make_data()
    results = {
        "FA + RoPE": run(FlashAttention, data),
        "FA (no pos)": run(FlashAttentionNoRoPE, data),
    }
    print_table(results)


if __name__ == "__main__":
    main()
