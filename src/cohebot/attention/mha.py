import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryPositionEmbedding


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention + RoPE.

    기존 CausalSelfAttention을 대체한다.
    절대 위치 임베딩 대신 RoPE를 Q, K에 적용한다.

    Args:
        embed_dim: 모델 임베딩 차원.
        num_heads: 어텐션 헤드 수.
        max_seq_len: RoPE 최대 시퀀스 길이.
        dropout: 어텐션 드롭아웃 비율.
        bias: 선형 레이어 bias 사용 여부.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:S, :S].unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        out = self.resid_dropout(self.out_proj(out))
        return out
