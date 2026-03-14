import torch
import torch.nn as nn

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
        # TODO:
        #   - Q, K, V 프로젝션 (embed_dim -> embed_dim)
        #   - 출력 프로젝션 (embed_dim -> embed_dim)
        #   - RotaryPositionEmbedding(head_dim, max_seq_len)
        #   - causal mask 등록 (register_buffer)
        #   - 어텐션 드롭아웃 + 잔차 드롭아웃
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            (batch, seq_len, embed_dim)
        """
        # TODO:
        #   1. Q, K, V 프로젝션
        #   2. (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        #   3. RoPE 적용: self.rope(q, k)
        #   4. 어텐션 스코어 계산 + causal mask
        #   5. softmax -> 어텐션 드롭아웃 -> 가중합
        #   6. 헤드 결합 -> 출력 프로젝션 -> 잔차 드롭아웃
        raise NotImplementedError
