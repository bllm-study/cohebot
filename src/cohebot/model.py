import math
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ff_dim: int = 3072
    dropout: float = 0.1
    bias: bool = True


class GELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.embed_dim = config.embed_dim

        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.max_seq_len, config.max_seq_len),
                diagonal=1
            ).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.embed_dim, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:seq_len, :seq_len],
            float("-inf")
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


class FeedForward(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.ff_dim, bias=config.bias)
        self.gelu = GELU()
        self.fc2 = nn.Linear(config.ff_dim, config.embed_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.embed_dim)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.ln_final = LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.tok_embed.weight = self.lm_head.weight

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"모델 파라미터 수: {n_params / 1e6:.2f}M")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        assert seq_len <= self.config.max_seq_len, \
            f"시퀀스 길이 {seq_len}이 최대 길이 {self.config.max_seq_len}을 초과합니다"

        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)

        tok_emb = self.tok_embed(input_ids)
        pos_emb = self.pos_embed(positions)
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str | torch.device = "cpu",
    ) -> "GPT":
        """체크포인트에서 모델을 로드합니다.

        Args:
            checkpoint_path: 체크포인트 파일 경로 (.pt 또는 .pth)
            device: 모델을 로드할 디바이스

        Returns:
            로드된 GPT 모델
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "config" in checkpoint:
            config = checkpoint["config"]
        else:
            config = GPTConfig()
            print("경고: 체크포인트에 config가 없어 기본 설정을 사용합니다.")

        model = cls(config)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        print(f"체크포인트 로드 완료: {checkpoint_path}")

        if "step" in checkpoint:
            print(f"  - 학습 스텝: {checkpoint['step']}")
        if "epoch" in checkpoint:
            print(f"  - 에포크: {checkpoint['epoch']}")
        if "loss" in checkpoint:
            print(f"  - 손실: {checkpoint['loss']:.4f}")

        return model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = input_ids
            if input_ids.size(1) > self.config.max_seq_len:
                idx_cond = input_ids[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


GPT2_SMALL = GPTConfig(
    vocab_size=50257,
    max_seq_len=1024,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    ff_dim=3072,
)

GPT2_MEDIUM = GPTConfig(
    vocab_size=50257,
    max_seq_len=1024,
    embed_dim=1024,
    num_heads=16,
    num_layers=24,
    ff_dim=4096,
)

GPT2_LARGE = GPTConfig(
    vocab_size=50257,
    max_seq_len=1024,
    embed_dim=1280,
    num_heads=20,
    num_layers=36,
    ff_dim=5120,
)

GPT2_TINY = GPTConfig(
    vocab_size=50257,
    max_seq_len=256,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    ff_dim=512,
    dropout=0.1,
)

GPT2_M3_PRO = GPTConfig(
    vocab_size=50257,
    max_seq_len=512,
    embed_dim=384,
    num_heads=6,
    num_layers=6,
    ff_dim=1536,
    dropout=0.1,
)

GPT2_M3_PRO_LARGE = GPTConfig(
    vocab_size=50257,
    max_seq_len=1024,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    ff_dim=3072,
    dropout=0.1,
)


if __name__ == "__main__":
    config = GPT2_TINY
    model = GPT(config)

    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(input_ids, targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    start_tokens = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(start_tokens, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
