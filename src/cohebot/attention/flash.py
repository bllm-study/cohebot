import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryPositionEmbedding


class FlashAttention(nn.Module):
    """Flash Attention + RoPE.

    F.scaled_dot_product_attention 기반.
    어텐션 행렬을 실체화하지 않아 O(N) 메모리.
    GQA 구조를 지원한다 (num_kv_heads < num_heads).

    Args:
        embed_dim: 모델 임베딩 차원.
        num_heads: Q 헤드 수.
        num_kv_heads: KV 헤드 수. num_heads와 같으면 MHA.
        max_seq_len: RoPE 최대 시퀀스 길이.
        dropout: 어텐션 드롭아웃 비율 (학습 시에만 적용).
        bias: 선형 레이어 bias 사용 여부.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        # TODO:
        #   - GQA와 동일한 Q/K/V 프로젝션 구조
        #   - RotaryPositionEmbedding(head_dim, max_seq_len)
        #   - 출력 프로젝션
        #   - causal mask 버퍼 불필요 (is_causal 파라미터 사용)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            (batch, seq_len, embed_dim)
        """
        # TODO:
        #   1. Q, K, V 프로젝션 (GQA 구조)
        #   2. reshape + transpose
        #   3. RoPE 적용: self.rope(q, k)
        #   4. KV 헤드 확장 (num_kv_heads != num_heads인 경우)
        #   5. F.scaled_dot_product_attention(q, k, v,
        #        dropout_p=..., is_causal=True)
        #   6. 헤드 결합 -> 출력 프로젝션 -> 잔차 드롭아웃
        raise NotImplementedError
