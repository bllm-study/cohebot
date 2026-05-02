#!/usr/bin/env python3
"""학습 진행 검증 스크립트.

맥북(MPS) / GPU 어디서든 동일한 명령으로 실행해서, 학습이 실제로 진행되는지
정량적으로 증명한다.

세 축으로 검증한다:
  1. 메트릭 — step별 loss / grad_norm / param-update L2 / perplexity / throughput
  2. 재현성 + 환경 — seed 고정, python/torch/device 정보, peak memory
  3. 실제 데이터 dry-run — bllm-study/kowiki-corpus 일부로 N step 학습이 도는지 확인

산출물: artifacts/verify/<timestamp>/{metrics.csv, summary.md, env.json, loss_curve.png}

사용법:
    # 합성 데이터 (기본, 외부 의존 없음)
    uv run python scripts/verify_training.py
    uv run python scripts/verify_training.py --preset 100m --steps 50

    # 실제 데이터 dry-run (HF Hub에서 일부 다운로드)
    uv run python scripts/verify_training.py --real-data --steps 5

    # TOML 설정 사용
    uv run python scripts/verify_training.py --config configs/smoke-100m.toml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cohebot.dataset import create_dataloader
from cohebot.model import CoheLLMBot, CoheLLMBotConfig
from cohebot.tokenizer import GPT2Tokenizer
from cohebot.train import _resolve_model_config, load_config

PRESETS: dict[str, CoheLLMBotConfig] = {
    "tiny": CoheLLMBotConfig(
        max_seq_len=128,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=512,
        dropout=0.0,
        attn_type="mha",
    ),
    "small": CoheLLMBotConfig(
        max_seq_len=256,
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        ff_dim=1536,
        dropout=0.0,
        attn_type="mha",
    ),
    "100m": CoheLLMBotConfig(
        max_seq_len=512,
        embed_dim=768,
        num_heads=12,
        num_layers=8,
        ff_dim=3072,
        dropout=0.0,
        attn_type="mha",
    ),
}

_SYNTHETIC_KOREAN = (
    "인공지능은 인간의 학습 능력과 추론 능력을 컴퓨터가 모방할 수 있도록 하는 기술이다. "
    "기계 학습은 인공지능의 한 분야로서 데이터로부터 패턴을 학습하여 예측을 수행한다. "
    "딥러닝은 인공 신경망을 여러 층으로 쌓아 복잡한 표현을 학습하는 방법이다. "
    "트랜스포머 모델은 어텐션 메커니즘을 기반으로 하여 시퀀스 데이터를 효과적으로 처리한다. "
    "대규모 언어 모델은 방대한 텍스트 데이터를 학습하여 자연어를 이해하고 생성할 수 있다. "
    "한국어 자연어 처리는 교착어의 특성을 고려한 형태소 분석이 중요한 전처리 과정이다. "
    "위키백과는 다양한 주제에 대한 백과사전적 지식을 담고 있는 대규모 텍스트 코퍼스이다. "
    "토크나이저는 텍스트를 모델이 처리할 수 있는 토큰 단위로 분할하는 역할을 한다. "
    "역전파 알고리즘은 신경망의 가중치를 업데이트하기 위해 손실 함수의 기울기를 계산한다. "
    "배치 정규화와 레이어 정규화는 학습을 안정화시키는 데 널리 사용되는 기법이다. "
)


@dataclass
class StepMetric:
    step: int
    loss: float
    lr: float
    grad_norm: float
    param_update_norm: float
    perplexity: float
    tokens_per_sec: float


def _detect_device(forced: str | None) -> torch.device:
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_corpus(tokenizer: GPT2Tokenizer, min_tokens: int) -> str:
    tokens_per_copy = len(tokenizer.encode(_SYNTHETIC_KOREAN))
    repeats = max(1, (min_tokens // tokens_per_copy) + 1)
    return _SYNTHETIC_KOREAN * repeats


def _load_real_corpus(min_chars: int, cache_dir: Path) -> tuple[str, dict]:
    """bllm-study/kowiki-corpus 일부를 받아서 코퍼스 텍스트로 반환.

    HF Hub에 파일이 있으면 받고, 없으면 datasets 스트리밍으로 폴백한다.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "bllm-study/kowiki-corpus"
    meta: dict = {"source": repo_id}

    try:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename="wiki_corpus.txt",
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        meta["path"] = local_path
        with open(local_path, encoding="utf-8") as f:
            text = f.read(min_chars * 4)
        return text, meta
    except Exception as hub_err:
        meta["hub_error"] = repr(hub_err)

    try:
        from datasets import load_dataset

        ds = load_dataset(repo_id, split="train", streaming=True)
        chunks: list[str] = []
        total = 0
        for row in ds:
            piece = row.get("text") or row.get("content") or ""
            if not piece:
                continue
            chunks.append(piece)
            total += len(piece)
            if total >= min_chars:
                break
        if chunks:
            meta["streamed_rows"] = len(chunks)
            return "\n".join(chunks), meta
    except Exception as ds_err:
        meta["datasets_error"] = repr(ds_err)

    raise RuntimeError(f"실제 데이터 로드 실패. HF Hub 접근 가능한지 확인하세요. meta={meta}")


def _env_report(device: torch.device) -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    elif device.type == "mps":
        info["mps_built"] = torch.backends.mps.is_built()
    return info


def _memory_report_mb(device: torch.device) -> tuple[float | None, str]:
    """디바이스별 메모리 사용량.

    CUDA: peak (max_memory_allocated). MPS: current (peak API 부재).
    어떤 의미인지 함께 반환한다.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6, "peak"
    if device.type == "mps":
        if hasattr(torch.mps, "driver_allocated_memory"):
            return torch.mps.driver_allocated_memory() / 1e6, "driver_current"
        if hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / 1e6, "current"
    return None, "n/a"


def run_verification(
    config: CoheLLMBotConfig,
    *,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    seed: int,
    learning_rate: float,
    out_dir: Path,
    use_real_data: bool,
) -> tuple[list[StepMetric], dict]:
    _set_seed(seed)
    config.dropout = 0.0

    tokenizer = GPT2Tokenizer()
    data_meta: dict = {"mode": "real" if use_real_data else "synthetic"}
    if use_real_data:
        min_chars = max(50_000, seq_len * batch_size * num_steps * 8)
        corpus, src_meta = _load_real_corpus(min_chars, cache_dir=out_dir.parent / "_cache")
        data_meta.update(src_meta)
        data_meta["chars"] = len(corpus)
    else:
        corpus = _build_corpus(tokenizer, seq_len * batch_size * max(64, num_steps * 2))
        data_meta["chars"] = len(corpus)

    loader = create_dataloader(
        corpus,
        tokenizer,
        batch_size=batch_size,
        max_length=seq_len,
        stride=seq_len // 2,
        shuffle=True,
    )
    data_meta["batches"] = len(loader)

    model = CoheLLMBot(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Δθ 측정용 sentinel: 첫 학습 가능 파라미터 하나만 추적.
    # 전체 clone은 100m 모델 기준 step당 ~400MB 추가 할당이라 지양.
    sentinel_param = next(p for p in model.parameters() if p.requires_grad)

    metrics: list[StepMetric] = []
    model.train()
    data_iter = iter(loader)
    total_tokens = 0
    t_total_start = time.time()

    for step in range(1, num_steps + 1):
        try:
            input_ids, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            input_ids, targets = next(data_iter)

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        prev_sentinel = sentinel_param.detach().clone()

        t0 = time.time()
        _, loss = model(input_ids, targets)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - t0

        param_update_norm = (sentinel_param.detach() - prev_sentinel).pow(2).sum().sqrt().item()

        loss_val = loss.item()
        tokens = batch_size * seq_len
        total_tokens += tokens
        metrics.append(
            StepMetric(
                step=step,
                loss=loss_val,
                lr=optimizer.param_groups[0]["lr"],
                grad_norm=grad_norm,
                param_update_norm=param_update_norm,
                perplexity=math.exp(min(loss_val, 20)),
                tokens_per_sec=tokens / max(elapsed, 1e-9),
            )
        )
        print(
            f"  step {step:>4d} | loss={loss_val:7.4f} | "
            f"grad_norm={grad_norm:6.3f} | Δθ={param_update_norm:7.4f} | "
            f"ppl={metrics[-1].perplexity:8.1f} | {metrics[-1].tokens_per_sec:6.0f} tok/s"
        )

    total_elapsed = time.time() - t_total_start
    mem_mb, mem_kind = _memory_report_mb(device)
    summary = {
        "total_elapsed_sec": total_elapsed,
        "avg_tokens_per_sec": total_tokens / max(total_elapsed, 1e-9),
        "memory_mb": mem_mb,
        "memory_kind": mem_kind,
        "n_params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "sentinel_param": next(
            (n for n, p in model.named_parameters() if p is sentinel_param), "?"
        ),
        "data": data_meta,
    }
    return metrics, summary


def _check_progression(metrics: list[StepMetric]) -> tuple[bool, list[str]]:
    """학습 진행 여부 판정. 모든 기준을 통과해야 PASS."""
    failures: list[str] = []
    losses = [m.loss for m in metrics]

    if not all(math.isfinite(loss) for loss in losses):
        failures.append("loss에 NaN/Inf 발생")
    if not all(math.isfinite(m.grad_norm) for m in metrics):
        failures.append("grad_norm에 NaN/Inf 발생")
    if not all(m.param_update_norm > 0 for m in metrics):
        failures.append("일부 step에서 파라미터가 갱신되지 않음 (Δθ=0)")

    mid = len(losses) // 2
    if mid > 0:
        first_half = sum(losses[:mid]) / mid
        second_half = sum(losses[mid:]) / (len(losses) - mid)
        if second_half >= first_half:
            failures.append(
                f"loss가 감소하지 않음 (전반부 {first_half:.4f} → 후반부 {second_half:.4f})"
            )

    return len(failures) == 0, failures


def _write_csv(path: Path, metrics: list[StepMetric]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "loss",
            "lr",
            "grad_norm",
            "param_update_norm",
            "perplexity",
            "tokens_per_sec",
        ])
        for m in metrics:
            writer.writerow([
                m.step,
                f"{m.loss:.6f}",
                f"{m.lr:.6e}",
                f"{m.grad_norm:.6f}",
                f"{m.param_update_norm:.6f}",
                f"{m.perplexity:.4f}",
                f"{m.tokens_per_sec:.2f}",
            ])


def _write_summary(
    path: Path,
    *,
    env: dict,
    config: CoheLLMBotConfig,
    metrics: list[StepMetric],
    summary: dict,
    passed: bool,
    failures: list[str],
    args: argparse.Namespace,
    plotted: bool,
) -> None:
    losses = [m.loss for m in metrics]
    mid = len(losses) // 2
    first_half = sum(losses[:mid]) / max(1, mid)
    second_half = sum(losses[mid:]) / max(1, len(losses) - mid)

    lines: list[str] = []
    lines.append(f"# 학습 진행 검증 리포트 — {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"**결과**: {'PASS' if passed else 'FAIL'}")
    if failures:
        lines.append("")
        lines.append("**실패 사유**:")
        for f in failures:
            lines.append(f"- {f}")
    lines.append("")
    lines.append("## 환경")
    for k, v in env.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## 실행 설정")
    if args.config:
        lines.append(f"- config: `{args.config}`")
    else:
        lines.append(f"- preset: `{args.preset}`")
    lines.append(f"- steps: {args.steps}, batch_size: {args.batch_size}, seq_len: {args.seq_len}")
    lines.append(f"- learning_rate: {args.lr}, seed: {args.seed}")
    data_meta = summary.get("data", {})
    lines.append(
        f"- 데이터 모드: `{data_meta.get('mode', '?')}`"
        + (
            f" ({data_meta.get('chars', 0):,} chars / {data_meta.get('batches', 0)} batches)"
            if data_meta
            else ""
        )
    )
    if data_meta.get("source"):
        lines.append(f"- 데이터 출처: `{data_meta['source']}`")
    lines.append("")
    lines.append("## 모델")
    lines.append(f"- 파라미터: {summary['n_params_m']:.2f}M")
    lines.append(
        f"- embed_dim={config.embed_dim}, layers={config.num_layers}, "
        f"heads={config.num_heads}, ff_dim={config.ff_dim}, attn={config.attn_type}"
    )
    lines.append("")
    lines.append("## 학습 진행 요약")
    lines.append(f"- 총 소요 시간: {summary['total_elapsed_sec']:.2f}초")
    lines.append(f"- 평균 throughput: {summary['avg_tokens_per_sec']:.0f} tokens/sec")
    if summary["memory_mb"] is not None:
        lines.append(f"- 메모리 ({summary['memory_kind']}): {summary['memory_mb']:.1f} MB")
    lines.append(f"- loss: {losses[0]:.4f} → {losses[-1]:.4f} (Δ={losses[-1] - losses[0]:+.4f})")
    lines.append(f"- 전반부 평균 loss: {first_half:.4f}")
    lines.append(f"- 후반부 평균 loss: {second_half:.4f}")
    lines.append(f"- perplexity: {metrics[0].perplexity:.1f} → {metrics[-1].perplexity:.1f}")
    lines.append(f"- Δθ sentinel 파라미터: `{summary.get('sentinel_param', '?')}`")
    lines.append("")
    lines.append("## 판정 기준")
    lines.append("- 모든 step의 loss / grad_norm 이 finite")
    lines.append("- 모든 step에서 sentinel 파라미터 업데이트 노름 Δθ > 0")
    lines.append("- 후반부 평균 loss < 전반부 평균 loss (실제로 줄어들고 있는가)")
    lines.append("")
    lines.append("## 산출물")
    lines.append("- `metrics.csv` — step별 메트릭 원본")
    lines.append("- `summary.md` — 본 리포트")
    lines.append("- `env.json` — 환경 정보 JSON")
    if plotted:
        lines.append("- `loss_curve.png` — loss/grad_norm/Δθ 시각화")

    path.write_text("\n".join(lines), encoding="utf-8")


def _maybe_plot(path: Path, metrics: list[StepMetric]) -> bool:
    try:
        import matplotlib  # pyright: ignore[reportMissingImports]

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False

    steps = [m.step for m in metrics]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(steps, [m.loss for m in metrics], color="tab:blue")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, [m.grad_norm for m in metrics], color="tab:orange")
    axes[1].set_ylabel("grad_norm")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps, [m.param_update_norm for m in metrics], color="tab:green")
    axes[2].set_ylabel("param update L2")
    axes[2].set_xlabel("step")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Training progression")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="학습 진행 검증 스크립트 (맥북 MPS / GPU 공통)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preset", choices=list(PRESETS), default="tiny")
    parser.add_argument("--config", type=str, default=None, help="TOML 설정 (preset 대신)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps")
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/verify",
        help="산출물 디렉토리 (timestamp 하위 디렉토리로 저장)",
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="bllm-study/kowiki-corpus에서 실제 데이터 일부를 받아 dry-run",
    )
    parser.add_argument(
        "--attn-type",
        choices=["mha", "gqa", "flash"],
        default=None,
        help="attention 종류 강제 지정 (preset/config 기본값을 덮어씀)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.config:
        cfg = load_config(args.config)
        config = _resolve_model_config(cfg)
        train_cfg = cfg.get("training", {})
        batch_size = args.batch_size or train_cfg.get("batch_size", 4)
        seq_len = args.seq_len or train_cfg.get("seq_len", 256)
    else:
        config = PRESETS[args.preset]
        batch_size = args.batch_size or (2 if args.preset == "tiny" else 4)
        seq_len = args.seq_len or min(config.max_seq_len, 128 if args.preset == "tiny" else 256)

    args.batch_size = batch_size
    args.seq_len = seq_len

    if args.attn_type:
        config.attn_type = args.attn_type

    device = _detect_device(args.device)
    env = _env_report(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 64}")
    print("Training verification")
    print(f"{'=' * 64}")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print(f"  preset: {args.preset}, steps: {args.steps}, batch×seq: {batch_size}×{seq_len}")
    print(f"  data: {'real (HF Hub)' if args.real_data else 'synthetic'}")
    print(f"  out: {out_dir}")
    print(f"{'=' * 64}\n")

    metrics, summary = run_verification(
        config,
        num_steps=args.steps,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        seed=args.seed,
        learning_rate=args.lr,
        out_dir=out_dir,
        use_real_data=args.real_data,
    )

    passed, failures = _check_progression(metrics)

    _write_csv(out_dir / "metrics.csv", metrics)
    plotted = _maybe_plot(out_dir / "loss_curve.png", metrics)
    _write_summary(
        out_dir / "summary.md",
        env=env,
        config=config,
        metrics=metrics,
        summary=summary,
        passed=passed,
        failures=failures,
        args=args,
        plotted=plotted,
    )

    (out_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 64}")
    print(f"  결과: {'PASS' if passed else 'FAIL'}")
    if failures:
        for f in failures:
            print(f"    - {f}")
    print(
        f"  소요 시간: {summary['total_elapsed_sec']:.2f}초, "
        f"throughput: {summary['avg_tokens_per_sec']:.0f} tok/s"
    )
    if summary["memory_mb"] is not None:
        print(f"  memory ({summary['memory_kind']}): {summary['memory_mb']:.1f} MB")
    print(f"  산출물: {out_dir}/")
    print(
        "    - metrics.csv, summary.md, env.json"
        + (", loss_curve.png" if plotted else " (matplotlib 미설치 시 PNG 생략)")
    )
    print(f"{'=' * 64}\n")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
