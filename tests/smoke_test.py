#!/usr/bin/env python3
"""CoheLLMBot smoke test — 외부 데이터 없이 학습 파이프라인 전체를 검증한다.

사용법:
    # 기본 (tiny 모델, ~7M, CPU에서 빠르게)
    uv run python tests/smoke_test.py

    # ~100M 모델 (GPU 권장)
    uv run python tests/smoke_test.py --preset 100m

    # TOML 설정 파일로 커스텀
    uv run python tests/smoke_test.py --config configs/smoke-100m.toml

    # 특정 스텝 수 / 디바이스 지정
    uv run python tests/smoke_test.py --steps 5 --device cpu
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from cohebot.dataset import create_dataloader
from cohebot.model import CoheLLMBot, CoheLLMBotConfig
from cohebot.tokenizer import GPT2Tokenizer
from cohebot.train import Trainer, load_config, _resolve_model_config


# ── 모델 프리셋 ─────────────────────────────────

PRESETS: dict[str, CoheLLMBotConfig] = {
    "tiny": CoheLLMBotConfig(
        max_seq_len=128, embed_dim=128, num_heads=4,
        num_layers=2, ff_dim=512, dropout=0.0, attn_type="mha",
    ),
    "small": CoheLLMBotConfig(
        max_seq_len=256, embed_dim=384, num_heads=6,
        num_layers=6, ff_dim=1536, dropout=0.0, attn_type="mha",
    ),
    "100m": CoheLLMBotConfig(
        max_seq_len=512, embed_dim=768, num_heads=12,
        num_layers=8, ff_dim=3072, dropout=0.0, attn_type="mha",
    ),
}

# ── 합성 데이터 ─────────────────────────────────

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


def _build_synthetic_corpus(tokenizer: GPT2Tokenizer, min_tokens: int) -> str:
    """학습에 충분한 양의 합성 코퍼스를 반복 생성한다."""
    tokens_per_copy = len(tokenizer.encode(_SYNTHETIC_KOREAN))
    repeats = max(1, (min_tokens // tokens_per_copy) + 1)
    return _SYNTHETIC_KOREAN * repeats


# ── 검증 결과 ───────────────────────────────────

@dataclass
class _TestResult:
    name: str
    passed: bool
    detail: str = ""
    elapsed_sec: float = 0.0


# ── 디바이스 감지 ───────────────────────────────

def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Smoke Test Runner ───────────────────────────

class SmokeTestRunner:
    """학습 파이프라인의 각 단계를 개별 검증한다."""

    def __init__(
        self,
        config: CoheLLMBotConfig,
        *,
        num_steps: int = 10,
        batch_size: int = 4,
        seq_len: int | None = None,
        device: str | None = None,
    ):
        self.config = config
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.seq_len = seq_len or min(config.max_seq_len, 128)
        self.device = torch.device(device) if device else _detect_device()
        self.tokenizer = GPT2Tokenizer()
        self.results: list[_TestResult] = []

    # ── 내부 유틸 ──

    def _record(self, name: str, passed: bool, detail: str = "", elapsed: float = 0.0):
        self.results.append(_TestResult(name, passed, detail, elapsed))
        status = "PASS" if passed else "FAIL"
        time_str = f" ({elapsed:.2f}s)" if elapsed > 0 else ""
        print(f"  [{status}] {name}{time_str}")
        if detail:
            print(f"         {detail}")

    def _make_loader(self, min_tokens: int, *, shuffle: bool = True):
        corpus = _build_synthetic_corpus(self.tokenizer, min_tokens)
        return create_dataloader(
            corpus, self.tokenizer,
            batch_size=self.batch_size,
            max_length=self.seq_len,
            stride=self.seq_len // 2,
            shuffle=shuffle,
        )

    # ── 개별 테스트 ──

    def test_model_creation(self) -> CoheLLMBot | None:
        """모델 인스턴스 생성 및 파라미터 수 검증."""
        t0 = time.time()
        try:
            model = CoheLLMBot(self.config)
            n_params = sum(p.numel() for p in model.parameters())
            elapsed = time.time() - t0
            self._record(
                "모델 생성", True,
                f"총 {n_params / 1e6:.2f}M params", elapsed,
            )
            return model
        except Exception as e:
            self._record("모델 생성", False, str(e), time.time() - t0)
            return None

    def test_forward(self, model: CoheLLMBot) -> bool:
        """Forward pass — logits shape과 loss 계산을 확인한다."""
        t0 = time.time()
        try:
            model.to(self.device)
            x = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len), device=self.device)
            y = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len), device=self.device)
            logits, loss = model(x, y)

            assert logits.shape == (self.batch_size, self.seq_len, self.config.vocab_size)
            assert loss is not None and loss.item() > 0

            elapsed = time.time() - t0
            self._record(
                "Forward pass", True,
                f"logits={list(logits.shape)}, loss={loss.item():.4f}", elapsed,
            )
            return True
        except Exception as e:
            self._record("Forward pass", False, str(e), time.time() - t0)
            return False

    def test_backward(self, model: CoheLLMBot) -> bool:
        """Backward pass — 모든 파라미터에 gradient가 생성되는지 확인한다."""
        t0 = time.time()
        try:
            x = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len), device=self.device)
            y = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len), device=self.device)
            _, loss = model(x, y)
            loss.backward()

            has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
            elapsed = time.time() - t0
            self._record("Backward pass", has_grad, f"all grads computed: {has_grad}", elapsed)
            model.zero_grad()
            return has_grad
        except Exception as e:
            self._record("Backward pass", False, str(e), time.time() - t0)
            return False

    def test_training_loop(self, model: CoheLLMBot) -> bool:
        """동일 배치를 반복 학습하여 loss가 감소(overfitting)하는지 확인한다."""
        t0 = time.time()
        try:
            loader = self._make_loader(self.seq_len * self.batch_size * 4, shuffle=False)
            input_ids, targets = next(iter(loader))
            input_ids, targets = input_ids.to(self.device), targets.to(self.device)

            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

            losses: list[float] = []
            for _ in range(self.num_steps):
                _, loss = model(input_ids, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

            elapsed = time.time() - t0
            first, last = losses[0], losses[-1]
            all_finite = all(math.isfinite(l) for l in losses)
            decreased = last < first
            throughput = (self.num_steps * self.batch_size * self.seq_len) / elapsed

            passed = all_finite and decreased
            detail = (
                f"loss: {first:.4f} → {last:.4f} "
                f"(Δ={last - first:+.4f}, {(1 - last / first) * 100:.1f}% 감소), "
                f"{throughput:.0f} tokens/sec"
            )
            if not decreased:
                detail += " — loss가 감소하지 않음"
            self._record("학습 루프 (loss 감소)", passed, detail, elapsed)
            return passed
        except Exception as e:
            self._record("학습 루프 (loss 감소)", False, str(e), time.time() - t0)
            return False

    def test_dataset_integration(self, model: CoheLLMBot) -> bool:
        """실제 토크나이저+DataLoader로 여러 배치를 순회하며 loss 추세를 확인한다."""
        t0 = time.time()
        try:
            loader = self._make_loader(self.seq_len * self.batch_size * 20)
            n_batches = len(loader)
            assert n_batches > 0, f"DataLoader가 비어 있음"

            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            losses: list[float] = []
            steps = min(self.num_steps, n_batches)

            for i, (input_ids, targets) in enumerate(loader):
                if i >= steps:
                    break
                input_ids, targets = input_ids.to(self.device), targets.to(self.device)
                _, loss = model(input_ids, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            elapsed = time.time() - t0
            all_finite = all(math.isfinite(l) for l in losses)
            mid = len(losses) // 2
            first_half = sum(losses[:mid]) / max(1, mid)
            second_half = sum(losses[mid:]) / max(1, len(losses) - mid)
            trend_down = second_half < first_half

            passed = all_finite and trend_down
            detail = (
                f"{steps}/{n_batches} batches, "
                f"loss: {losses[0]:.4f} → {losses[-1]:.4f}, "
                f"avg 전반부={first_half:.4f} → 후반부={second_half:.4f}"
            )
            if not trend_down:
                detail += " — loss 추세가 감소하지 않음"
            self._record("데이터셋 통합 (loss 추세)", passed, detail, elapsed)
            return passed
        except Exception as e:
            self._record("데이터셋 통합 (loss 추세)", False, str(e), time.time() - t0)
            return False

    def test_generation(self, model: CoheLLMBot) -> bool:
        """프롬프트로부터 토큰을 생성할 수 있는지 확인한다."""
        t0 = time.time()
        try:
            model.eval()
            prompt = "인공지능은"
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            generated = model.generate(input_ids, max_new_tokens=20, temperature=0.8, top_k=50)
            output_text = self.tokenizer.decode(generated[0].tolist())

            elapsed = time.time() - t0
            passed = generated.shape[1] > input_ids.shape[1]
            self._record(
                "텍스트 생성", passed,
                f"'{output_text[:80]}...' ({generated.shape[1]} tokens)", elapsed,
            )
            return passed
        except Exception as e:
            self._record("텍스트 생성", False, str(e), time.time() - t0)
            return False

    def test_checkpoint(self, model: CoheLLMBot) -> bool:
        """체크포인트 저장 → 로드 후 가중치가 일치하는지 확인한다."""
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_path = Path(tmpdir) / "smoke_checkpoint.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": model.config,
                    "epoch": 0, "step": 10, "loss": 5.0,
                }, ckpt_path)

                ckpt_size_mb = ckpt_path.stat().st_size / 1e6
                loaded = CoheLLMBot.from_checkpoint(str(ckpt_path), device=str(self.device))

                for (n1, p1), (n2, p2) in zip(
                    model.named_parameters(), loaded.named_parameters(),
                ):
                    assert n1 == n2, f"파라미터 이름 불일치: {n1} vs {n2}"
                    assert torch.equal(p1, p2), f"파라미터 값 불일치: {n1}"

            elapsed = time.time() - t0
            self._record("체크포인트 저장/로드", True, f"{ckpt_size_mb:.1f}MB", elapsed)
            return True
        except Exception as e:
            self._record("체크포인트 저장/로드", False, str(e), time.time() - t0)
            return False

    def test_trainer_integration(self, model: CoheLLMBot) -> bool:
        """실제 Trainer로 1에폭 학습 후 체크포인트에서 재개할 수 있는지 확인한다."""
        t0 = time.time()
        try:
            corpus = _build_synthetic_corpus(self.tokenizer, self.seq_len * self.batch_size * 30)
            split = int(len(corpus) * 0.8)
            train_loader = create_dataloader(
                corpus[:split], self.tokenizer,
                batch_size=self.batch_size, max_length=self.seq_len,
                stride=self.seq_len // 2, shuffle=True,
            )
            val_loader = create_dataloader(
                corpus[split:], self.tokenizer,
                batch_size=self.batch_size, max_length=self.seq_len,
                stride=self.seq_len // 2, shuffle=False,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = Trainer(
                    model=model, train_dataloader=train_loader, val_dataloader=val_loader,
                    learning_rate=1e-3, num_epochs=1, warmup_steps=2,
                    checkpoint_dir=tmpdir, device=str(self.device),
                    save_every_n_batches=0,
                )
                trainer.train(start_epoch=0)

                ckpt_path = Path(tmpdir) / "checkpoint_latest.pt"
                assert ckpt_path.exists(), "체크포인트 파일이 생성되지 않음"

                # 체크포인트에서 재개
                model2 = CoheLLMBot(self.config)
                trainer2 = Trainer(
                    model=model2, train_dataloader=train_loader, val_dataloader=val_loader,
                    learning_rate=1e-3, num_epochs=2, warmup_steps=2,
                    checkpoint_dir=tmpdir, device=str(self.device),
                    save_every_n_batches=0,
                )
                loaded_epoch, _ = trainer2.load_checkpoint(str(ckpt_path))

            elapsed = time.time() - t0
            self._record(
                "Trainer 통합 (학습+재개)", True,
                f"1에폭 학습 완료, 체크포인트 재개 성공 (epoch {loaded_epoch})", elapsed,
            )
            return True
        except Exception as e:
            self._record("Trainer 통합 (학습+재개)", False, str(e), time.time() - t0)
            return False

    def test_memory(self, model: CoheLLMBot) -> bool:
        """모델 메모리 사용량을 리포트한다."""
        t0 = time.time()
        try:
            param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
            total_mb = (param_bytes + buffer_bytes) / 1e6

            detail = f"모델: {total_mb:.1f}MB (params: {param_bytes / 1e6:.1f}MB, buffers: {buffer_bytes / 1e6:.1f}MB)"
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                detail += f", GPU: {torch.cuda.memory_allocated() / 1e6:.1f}MB allocated"

            elapsed = time.time() - t0
            self._record("메모리 사용량", True, detail, elapsed)
            return True
        except Exception as e:
            self._record("메모리 사용량", False, str(e), time.time() - t0)
            return False

    # ── 전체 실행 ──

    def run_all(self) -> bool:
        """모든 테스트를 순차 실행하고 결과를 요약한다."""
        print(f"\n{'=' * 60}")
        print("CoheLLMBot Smoke Test")
        print(f"{'=' * 60}")
        print(f"  Device:  {self.device}")
        print(f"  Model:   embed_dim={self.config.embed_dim}, "
              f"layers={self.config.num_layers}, heads={self.config.num_heads}")
        print(f"  Batch:   {self.batch_size} × {self.seq_len} tokens")
        print(f"  Steps:   {self.num_steps}")
        print(f"{'=' * 60}\n")

        model = self.test_model_creation()
        if model is None:
            print("\n모델 생성 실패 — 나머지 테스트를 건너뜁니다.")
            return False

        self.test_forward(model)
        self.test_backward(model)
        self.test_training_loop(model)
        self.test_dataset_integration(model)
        self.test_generation(model)
        self.test_checkpoint(model)
        self.test_trainer_integration(model)
        self.test_memory(model)

        del model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # 요약
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        total_time = sum(r.elapsed_sec for r in self.results)

        print(f"\n{'=' * 60}")
        print(f"결과: {passed}/{total} 통과 (총 {total_time:.2f}초)")
        if passed == total:
            print("모든 테스트 통과!")
        else:
            print("실패:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.detail}")
        print(f"{'=' * 60}\n")

        return passed == total


# ── CLI ──────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoheLLMBot smoke test — 외부 데이터 없이 학습 파이프라인 검증",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
프리셋:
  tiny   ~7M params   (CPU에서 수초 내 완료)
  small  ~30M params  (CPU 가능, GPU 권장)
  100m   ~95M params  (GPU 권장)

예시:
  uv run python tests/smoke_test.py                        # tiny (기본)
  uv run python tests/smoke_test.py --preset 100m          # ~100M
  uv run python tests/smoke_test.py --config configs/smoke-100m.toml
  uv run python tests/smoke_test.py --preset small --steps 20 --device cpu
""",
    )
    parser.add_argument("--preset", choices=list(PRESETS), default="tiny",
                        help="모델 프리셋 (기본: tiny)")
    parser.add_argument("--config", type=str, default=None,
                        help="TOML 설정 파일 (프리셋 대신 사용)")
    parser.add_argument("--steps", type=int, default=10, help="학습 스텝 수 (기본: 10)")
    parser.add_argument("--batch-size", type=int, default=None, help="배치 크기")
    parser.add_argument("--seq-len", type=int, default=None, help="시퀀스 길이")
    parser.add_argument("--device", type=str, default=None,
                        help="디바이스 강제 지정 (cpu, cuda, mps)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> bool:
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

    # smoke test에서는 dropout 0으로 고정 (재현성)
    config.dropout = 0.0

    runner = SmokeTestRunner(
        config, num_steps=args.steps, batch_size=batch_size,
        seq_len=seq_len, device=args.device,
    )
    return runner.run_all()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
