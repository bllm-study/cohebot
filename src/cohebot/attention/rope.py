import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Q, K에 회전 행렬을 적용하여 상대적 위치 정보를 인코딩한다.

    Args:
        dim: 헤드 차원 (embed_dim // num_heads). 짝수여야 한다.
        max_seq_len: 미리 계산할 최대 시퀀스 길이.
        base: 주파수 베이스 (기본 10000.0).
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("_cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("_sin_cached", freqs.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Q, K에 rotary embedding을 적용한다.

        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_kv_heads, seq_len, head_dim)
            offset: KV cache 사용 시 위치 오프셋.

        Returns:
            rotary embedding이 적용된 (q, k) 튜플.
        """
        seq_len = q.shape[2]
        cos = self._cos_cached[offset : offset + seq_len]
        sin = self._sin_cached[offset : offset + seq_len]

        def _rotate(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2)

        return _rotate(q), _rotate(k)
