import math
import time

import torch

from cohebot.model import CoheLLMBot, CoheLLMBotConfig

STEPS = 500
SEQ_LEN = 1024
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
CONFIG = CoheLLMBotConfig(
    vocab_size=50257, max_seq_len=SEQ_LEN, embed_dim=128,
    num_heads=4, num_layers=4, ff_dim=512, dropout=0.0,
)


def make_data(config=CONFIG):
    seq = torch.arange(config.max_seq_len + 1, device=DEVICE).unsqueeze(0)
    return seq[:, :-1], seq[:, 1:]


def run(attn_cls, data, config=CONFIG, steps=STEPS):
    torch.manual_seed(42)
    model = CoheLLMBot(config)
    for block in model.blocks:
        block.attn = attn_cls(
            embed_dim=config.embed_dim, num_heads=config.num_heads,
            max_seq_len=config.max_seq_len, dropout=config.dropout, bias=config.bias,
        )
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    losses = []
    t0 = time.time()
    for _ in range(steps):
        logits, loss = model(data[0], data[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses, time.time() - t0


def print_table(results, steps=STEPS):
    mid = steps // 2
    print(f"\n[device={DEVICE}, seq_len={SEQ_LEN}, steps={steps}]")
    print(f"{'Variant':<20} {'Step 1':>8} {f'Step {mid}':>9} {f'Step {steps}':>9} {'PPL':>8} {'Time':>7}")
    print("-" * 63)
    for name, (losses, t) in results.items():
        ppl = math.exp(min(losses[-1], 20))
        print(f"{name:<20} {losses[0]:>8.4f} {losses[mid-1]:>9.4f} {losses[-1]:>9.4f} {ppl:>8.2f} {t:>6.1f}s")
