import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryPositionEmbedding


class FlashAttention(nn.Module):
    """Flash Attention + RoPE.

    F.scaled_dot_product_attention 기반.
    어텐션 행렬을 실체화하지 않아 O(N) 메모리.

    Args:
        embed_dim: 모델 임베딩 차원.
        num_heads: 어텐션 헤드 수.
        max_seq_len: RoPE 최대 시퀀스 길이.
        dropout: 어텐션 드롭아웃 비율 (학습 시에만 적용).
        bias: 선형 레이어 bias 사용 여부.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})"
        )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        # Q/K/V 프로젝션 (embed_dim -> embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 출력 프로젝션
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # RotaryPositionEmbedding(head_dim, max_seq_len)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

        # 잔차 드롭아웃 — causal mask 버퍼 불필요 (is_causal 파라미터 사용)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            (batch, seq_len, embed_dim)
        """
        B, T, _ = x.shape

        # 1. Q, K, V 프로젝션
        # 2. reshape + transpose -> (B, num_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. RoPE 적용: self.rope(q, k)
        q, k = self.rope(q, k)

        # 4. F.scaled_dot_product_attention(q, k, v, dropout_p=..., is_causal=True)
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        # 5. 헤드 결합 -> 출력 프로젝션 -> 잔차 드롭아웃
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        out = self.resid_dropout(self.out_proj(out))

        return out
