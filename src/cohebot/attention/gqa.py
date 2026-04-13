import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention + RoPE.

    KV 헤드를 Q 헤드보다 적게 두어 메모리와 연산을 절약한다.
    num_kv_heads == num_heads이면 MHA, num_kv_heads == 1이면 MQA와 동일.

    Args:
        embed_dim: 모델 임베딩 차원.
        num_heads: Q 헤드 수.
        num_kv_heads: KV 헤드 수. num_heads의 약수여야 한다.
        max_seq_len: RoPE 최대 시퀀스 길이.
        dropout: 어텐션 드롭아웃 비율.
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
        #   - head_dim = embed_dim // num_heads
        #   - num_kv_groups = num_heads // num_kv_heads
        #   - Q 프로젝션: embed_dim -> num_heads * head_dim
        #   - K 프로젝션: embed_dim -> num_kv_heads * head_dim
        #   - V 프로젝션: embed_dim -> num_kv_heads * head_dim
        #   - 출력 프로젝션: embed_dim -> embed_dim
        #   - RotaryPositionEmbedding(head_dim, max_seq_len)
        #   - causal mask, 어텐션 드롭아웃 + 잔차 드롭아웃
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            (batch, seq_len, embed_dim)
        """
        # TODO:
        #   1. Q -> (batch, num_heads, seq, head_dim)
        #   2. K -> (batch, num_kv_heads, seq, head_dim)
        #   3. V -> (batch, num_kv_heads, seq, head_dim)
        #   4. RoPE 적용: self.rope(q, k)
        #   5. KV 헤드 확장: repeat_interleave(num_kv_groups)
        #   6. 어텐션 스코어 계산 + causal mask
        #   7. softmax -> 어텐션 드롭아웃 -> 가중합
        #   8. 헤드 결합 -> 출력 프로젝션 -> 잔차 드롭아웃
        raise NotImplementedError
