import math
import os
import sys
import time
from pathlib import Path
from typing import cast

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        tomllib = None

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .dataset import TextDataset, create_dataloader
from .model import CoheLLMBot, CoheLLMBotConfig
from .tokenizer import GPT2Tokenizer

try:
    import wandb  # type: ignore[import-untyped]

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _is_main_process() -> bool:
    """DDP 환경에서 rank 0인지 확인한다."""
    return int(os.environ.get("RANK", 0)) == 0


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _is_ddp() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


class Trainer:
    def __init__(
        self,
        model: CoheLLMBot,
        train_dataloader,
        val_dataloader=None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        num_epochs: int = 10,
        warmup_steps: int = 100,
        checkpoint_dir: str = "checkpoints",
        device: str = "auto",
        use_fp16: bool = False,
        save_every_n_batches: int = 1000,
        use_wandb: bool = False,
        wandb_project: str = "cohebot",
        wandb_run_name: str | None = None,
        upload_repo: str | None = None,
    ):
        # --- DDP 설정 ---
        self.ddp = _is_ddp()
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = _get_local_rank()
        self.is_main = _is_main_process()

        if self.ddp:
            dist.init_process_group("nccl")
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
            if self.is_main:
                print(f"DDP 활성화: {dist.get_world_size()} GPU")
        elif device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if self.is_main:
            print(f"학습 디바이스: {self.device}")

        self.model = model.to(self.device)
        self.use_fp16 = use_fp16
        if use_fp16:
            self.model = self.model.half()
            if self.is_main:
                print("FP16 모드 활성화")

        # DDP 래핑
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        self.raw_model: CoheLLMBot = cast(CoheLLMBot, self.model.module if self.ddp else self.model)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        decay_params = []
        no_decay_params = []

        for name, param in self.raw_model.named_parameters():
            if param.requires_grad:
                if "ln" in name or "bias" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=(0.9, 0.95),
        )

        total_steps = len(train_dataloader) * num_epochs
        steps_after_warmup = max(1, total_steps - warmup_steps)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=steps_after_warmup)

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.base_lr = learning_rate
        self.save_every_n_batches = save_every_n_batches
        self.start_batch_idx = 0

        # --- W&B ---
        self.use_wandb = use_wandb and self.is_main
        if self.use_wandb:
            if not _WANDB_AVAILABLE:
                print("wandb가 설치되지 않았습니다. pip install wandb")
                self.use_wandb = False
            else:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "model": vars(self.raw_model.config),
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "num_epochs": num_epochs,
                        "warmup_steps": warmup_steps,
                        "batch_size": train_dataloader.batch_size,
                        "fp16": use_fp16,
                        "ddp": self.ddp,
                        "world_size": dist.get_world_size() if self.ddp else 1,
                    },
                )
                print(f"W&B 로깅 활성화: {wandb_project}")

        # --- HF Hub 업로드 ---
        self.upload_repo = upload_repo

    def _log(self, metrics: dict, step: int | None = None):
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        processed_batches = 0

        # DDP: 에폭마다 sampler 시드 변경
        if self.ddp and hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(epoch)

        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            initial=self.start_batch_idx,
            total=num_batches,
            disable=not self.is_main,
        )

        for batch_idx, (input_ids, targets) in enumerate(pbar):
            if batch_idx < self.start_batch_idx:
                continue

            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            logits, loss = self.model(input_ids, targets)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if self.global_step < self.warmup_steps:
                lr = self.base_lr * (self.global_step + 1) / self.warmup_steps
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            self.optimizer.step()

            if self.global_step >= self.warmup_steps:
                self.scheduler.step()

            self.global_step += 1
            total_loss += loss.item()
            processed_batches += 1

            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.is_main:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})
                self._log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": current_lr,
                        "train/perplexity": math.exp(min(loss.item(), 20)),
                    },
                    step=self.global_step,
                )

            if self.save_every_n_batches > 0 and (batch_idx + 1) % self.save_every_n_batches == 0:
                if self.is_main:
                    self._save_mid_epoch_checkpoint(
                        epoch, batch_idx + 1, total_loss / processed_batches
                    )

        self.start_batch_idx = 0
        return total_loss / max(1, processed_batches)

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_dataloader is None:
            return 0.0

        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_dataloader)

        for input_ids, targets in self.val_dataloader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            logits, loss = self.model(input_ids, targets)
            total_loss += loss.item()

        return total_loss / num_batches

    def _save_mid_epoch_checkpoint(self, epoch: int, batch_idx: int, avg_loss: float):
        checkpoint = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "avg_loss": avg_loss,
            "config": self.raw_model.config,
            "is_mid_epoch": True,
        }

        path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, path)
        print(f"\n중간 체크포인트 저장: {path} (epoch {epoch + 1}, batch {batch_idx})")
        self._upload_checkpoint(path)

    def save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint = {
            "epoch": epoch,
            "batch_idx": 0,
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "val_loss": val_loss,
            "config": self.raw_model.config,
            "is_mid_epoch": False,
        }

        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"체크포인트 저장: {path}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best 모델 저장: {best_path}")

        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        self._upload_checkpoint(latest_path)

    def _upload_checkpoint(self, path: Path):
        """체크포인트를 HuggingFace Hub에 업로드한다."""
        if not self.upload_repo:
            return
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=self.upload_repo,
                repo_type="model",
            )
            print(f"HF Hub 업로드 완료: {self.upload_repo}/{path.name}")
        except Exception as e:
            print(f"HF Hub 업로드 실패: {e}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]

        is_mid_epoch = checkpoint.get("is_mid_epoch", False)
        batch_idx = checkpoint.get("batch_idx", 0)

        if is_mid_epoch:
            self.start_batch_idx = batch_idx
            print(f"체크포인트 로드: {path}")
            print(f"  에폭 {checkpoint['epoch'] + 1}, 배치 {batch_idx}부터 재개")
        else:
            self.start_batch_idx = 0
            print(f"체크포인트 로드: {path}")
            print(f"  에폭 {checkpoint['epoch'] + 1} 완료 상태에서 재개")

        return checkpoint["epoch"], is_mid_epoch

    def train(self, start_epoch: int = 0):
        if self.is_main:
            print(f"훈련 시작 (총 {self.num_epochs} 에폭)")
            print(f"배치 수: {len(self.train_dataloader)}")
            if self.save_every_n_batches > 0:
                print(f"체크포인트 저장 주기: {self.save_every_n_batches} 배치마다")

        try:
            for epoch in range(start_epoch, self.num_epochs):
                start_time = time.time()

                train_loss = self.train_epoch(epoch)

                val_loss = self.evaluate()

                epoch_time = time.time() - start_time

                if self.is_main:
                    print(f"\n에폭 {epoch + 1} 완료")
                    print(f"  훈련 손실: {train_loss:.4f}")
                    if self.val_dataloader:
                        print(f"  검증 손실: {val_loss:.4f}")
                        print(f"  Perplexity: {math.exp(val_loss):.2f}")
                    print(f"  소요 시간: {epoch_time:.1f}초\n")

                    self._log(
                        {
                            "epoch/train_loss": train_loss,
                            "epoch/val_loss": val_loss,
                            "epoch/perplexity": math.exp(val_loss) if val_loss > 0 else 0,
                            "epoch/time_sec": epoch_time,
                        },
                        step=self.global_step,
                    )

                    self.save_checkpoint(epoch, val_loss if self.val_dataloader else train_loss)

            if self.is_main:
                print("훈련 완료!")
                if self.use_wandb:
                    wandb.finish()

        except KeyboardInterrupt:
            if self.is_main:
                print("\n\n훈련 중단됨! 현재 상태 저장 중...")
                emergency_checkpoint = {
                    "epoch": epoch,
                    "batch_idx": self.global_step % len(self.train_dataloader),
                    "model_state_dict": self.raw_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "global_step": self.global_step,
                    "config": self.raw_model.config,
                    "is_mid_epoch": True,
                }
                path = self.checkpoint_dir / "checkpoint_latest.pt"
                torch.save(emergency_checkpoint, path)
                print(f"긴급 체크포인트 저장: {path}")
                self._upload_checkpoint(path)
                if self.use_wandb:
                    wandb.finish()
        finally:
            if self.ddp:
                dist.destroy_process_group()


def load_wikipedia_corpus(data_dir: str = "data") -> str:
    corpus_path = Path(data_dir) / "wiki_corpus.txt"

    if corpus_path.exists():
        print(f"기존 코퍼스 로드: {corpus_path}")
        with open(corpus_path, encoding="utf-8") as f:
            return f.read()

    raise FileNotFoundError(
        f"코퍼스 파일이 없습니다: {corpus_path}\n"
        f"다음 명령어로 생성하세요:\n"
        f"  uv run cohebot-crawl --data-dir {data_dir}\n"
        f"또는 HuggingFace Hub에서 다운로드하세요:\n"
        f"  uv run hf download bllm-study/kowiki-corpus --repo-type dataset --include 'wiki_corpus.txt' --local-dir {data_dir}"
    )


def load_config(path: str) -> dict:
    """TOML 설정 파일을 로드한다."""
    if tomllib is None:
        print("TOML 지원이 필요합니다. Python 3.11+ 이거나 tomli를 설치하세요:")
        print("  pip install tomli")
        sys.exit(1)

    with open(path, "rb") as f:
        return tomllib.load(f)


def _resolve_model_config(cfg: dict) -> CoheLLMBotConfig:
    """설정의 [model] 섹션에서 CoheLLMBotConfig를 생성한다."""
    model_cfg = cfg.get("model", {})

    return CoheLLMBotConfig(
        vocab_size=model_cfg.get("vocab_size", 50257),
        max_seq_len=model_cfg.get("max_seq_len", 1024),
        embed_dim=model_cfg.get("embed_dim", 768),
        num_heads=model_cfg.get("num_heads", 12),
        num_kv_heads=model_cfg.get("num_kv_heads"),
        num_layers=model_cfg.get("num_layers", 12),
        ff_dim=model_cfg.get("ff_dim", 3072),
        dropout=model_cfg.get("dropout", 0.1),
        bias=model_cfg.get("bias", True),
        attn_type=model_cfg.get("attn_type", "flash"),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CoheLLMBot 훈련")
    parser.add_argument(
        "--config", type=str, default=None, help="TOML 설정 파일 경로 (예: configs/cloud.toml)"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="배치 크기")
    parser.add_argument("--seq-len", type=int, default=None, help="시퀀스 길이")
    parser.add_argument("--epochs", type=int, default=None, help="에폭 수")
    parser.add_argument("--lr", type=float, default=None, help="학습률")
    parser.add_argument("--data-dir", type=str, default=None, help="데이터 디렉토리")
    parser.add_argument("--resume", type=str, default=None, help="체크포인트 경로")
    parser.add_argument("--max-chars", type=int, default=None, help="최대 코퍼스 문자 수 (샘플링)")
    parser.add_argument("--fp16", action="store_true", default=None, help="FP16 혼합 정밀도 사용")
    parser.add_argument(
        "--save-every", type=int, default=None, help="N 배치마다 체크포인트 저장 (0이면 비활성화)"
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="체크포인트 저장 디렉토리")
    args = parser.parse_args()

    # TOML 설정 로드
    if not args.config:
        print("--config 옵션으로 TOML 설정 파일을 지정하세요.")
        print("  예: cohebot-train --config configs/default.toml")
        sys.exit(1)

    cfg = load_config(args.config)
    if _is_main_process():
        print(f"설정 파일 로드: {args.config}")

    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    log_cfg = cfg.get("logging", {})

    # CLI 인자 > TOML 설정 > 기본값
    BATCH_SIZE = args.batch_size or train_cfg.get("batch_size", 4)
    MAX_SEQ_LEN = args.seq_len or train_cfg.get("seq_len", 512)
    NUM_EPOCHS = args.epochs or train_cfg.get("epochs", 3)
    LEARNING_RATE = args.lr or train_cfg.get("lr", 3e-4)
    USE_FP16 = args.fp16 if args.fp16 is not None else train_cfg.get("fp16", False)
    SAVE_EVERY = (
        args.save_every if args.save_every is not None else train_cfg.get("save_every", 1000)
    )
    WEIGHT_DECAY = train_cfg.get("weight_decay", 0.1)
    WARMUP_STEPS = train_cfg.get("warmup_steps", 100)
    MAX_GRAD_NORM = train_cfg.get("max_grad_norm", 1.0)

    DATA_DIR = args.data_dir or data_cfg.get("dir", "data")
    MAX_CHARS = args.max_chars or data_cfg.get("max_chars")
    CHECKPOINT_DIR = args.checkpoint_dir or ckpt_cfg.get("dir", "checkpoints")
    RESUME = args.resume or ckpt_cfg.get("resume")
    UPLOAD_REPO = ckpt_cfg.get("upload_repo")

    USE_WANDB = log_cfg.get("wandb", False)
    WANDB_PROJECT = log_cfg.get("wandb_project", "cohebot")
    WANDB_RUN_NAME = log_cfg.get("wandb_run_name")

    if _is_main_process():
        print("=" * 50)
        print("CoheLLMBot 훈련 설정")
        print("=" * 50)

    model_config = _resolve_model_config(cfg)
    model_config.max_seq_len = MAX_SEQ_LEN

    if _is_main_process():
        print(f"  설정 파일: {args.config}")
        print(f"  배치 크기: {BATCH_SIZE}")
        print(f"  시퀀스 길이: {MAX_SEQ_LEN}")
        print(f"  에폭 수: {NUM_EPOCHS}")
        print(f"  학습률: {LEARNING_RATE}")
        print(f"  FP16: {USE_FP16}")
        print(f"  체크포인트 저장 주기: {SAVE_EVERY} 배치")
        print(f"  체크포인트 디렉토리: {CHECKPOINT_DIR}")
        if UPLOAD_REPO:
            print(f"  HF Hub 업로드: {UPLOAD_REPO}")
        if USE_WANDB:
            print(f"  W&B 프로젝트: {WANDB_PROJECT}")
        print("=" * 50)

    tokenizer = GPT2Tokenizer()

    corpus = load_wikipedia_corpus(DATA_DIR)
    if _is_main_process():
        print(f"코퍼스 크기: {len(corpus):,} 문자 ({len(corpus) / 1e6:.1f}MB)")

    if MAX_CHARS and len(corpus) > MAX_CHARS:
        corpus = corpus[:MAX_CHARS]
        if _is_main_process():
            print(f"샘플링 후: {len(corpus):,} 문자 ({len(corpus) / 1e6:.1f}MB)")

    split_idx = int(len(corpus) * 0.95)
    train_text = corpus[:split_idx]
    val_text = corpus[split_idx:]

    if _is_main_process():
        print(f"훈련 데이터: {len(train_text):,} 문자")
        print(f"검증 데이터: {len(val_text):,} 문자")

    # DDP: DistributedSampler 사용
    if _is_ddp():
        train_dataset = TextDataset(train_text, tokenizer, MAX_SEQ_LEN, stride=MAX_SEQ_LEN // 2)
        val_dataset = TextDataset(val_text, tokenizer, MAX_SEQ_LEN, stride=MAX_SEQ_LEN // 2)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            sampler=val_sampler,
            drop_last=True,
        )
    else:
        train_loader = create_dataloader(
            train_text,
            tokenizer,
            batch_size=BATCH_SIZE,
            max_length=MAX_SEQ_LEN,
            stride=MAX_SEQ_LEN // 2,
            shuffle=True,
        )
        val_loader = create_dataloader(
            val_text,
            tokenizer,
            batch_size=BATCH_SIZE,
            max_length=MAX_SEQ_LEN,
            stride=MAX_SEQ_LEN // 2,
            shuffle=False,
        )

    if _is_main_process():
        print(f"훈련 배치 수: {len(train_loader):,}")
        print(f"검증 배치 수: {len(val_loader):,}")

    model = CoheLLMBot(model_config)

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        num_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        checkpoint_dir=CHECKPOINT_DIR,
        use_fp16=USE_FP16,
        save_every_n_batches=SAVE_EVERY,
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
        upload_repo=UPLOAD_REPO,
    )

    start_epoch = 0
    if RESUME:
        loaded_epoch, is_mid_epoch = trainer.load_checkpoint(RESUME)
        if is_mid_epoch:
            start_epoch = loaded_epoch
        else:
            start_epoch = loaded_epoch + 1

    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
