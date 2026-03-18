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
        # TODO: 주파수 텐서 사전 계산
        #   - theta_i = base^(-2i/dim), i = 0..dim//2-1
        #   - (max_seq_len, dim//2) 크기 freqs 생성
        #   - cos, sin 캐싱 (register_buffer)
        raise NotImplementedError

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
        # TODO:
        #   1. q, k를 (..., dim//2, 2)로 reshape하여 짝/홀수 분리
        #   2. 캐싱된 cos, sin에서 [offset:offset+seq_len] 슬라이스
        #   3. 회전 적용: even * cos - odd * sin, odd * cos + even * sin
        #   4. 원래 shape으로 복원
        raise NotImplementedError
