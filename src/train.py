import os
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import GPT, GPTConfig, GPT2_TINY, GPT2_SMALL, GPT2_M3_PRO
from tokenizer import GPT2Tokenizer
from dataset import create_dataloader


class Trainer:

    def __init__(
        self,
        model: GPT,
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
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"학습 디바이스: {self.device}")

        self.model = model.to(self.device)
        self.use_fp16 = use_fp16
        if use_fp16:
            self.model = self.model.half()
            print("FP16 모드 활성화")
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if "ln" in name or "bias" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        self.optimizer = AdamW([
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=learning_rate, betas=(0.9, 0.95))

        total_steps = len(train_dataloader) * num_epochs
        steps_after_warmup = max(1, total_steps - warmup_steps)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=steps_after_warmup)

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.base_lr = learning_rate
        self.save_every_n_batches = save_every_n_batches
        self.start_batch_idx = 0

    def _warmup_lr(self) -> float:
        if self.global_step < self.warmup_steps:
            return self.global_step / max(1, self.warmup_steps)
        return 1.0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        processed_batches = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                    initial=self.start_batch_idx, total=num_batches)

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

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            if self.save_every_n_batches > 0 and (batch_idx + 1) % self.save_every_n_batches == 0:
                self._save_mid_epoch_checkpoint(epoch, batch_idx + 1, total_loss / processed_batches)

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
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "avg_loss": avg_loss,
            "config": self.model.config,
            "is_mid_epoch": True,
        }

        path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, path)
        print(f"\n중간 체크포인트 저장: {path} (epoch {epoch+1}, batch {batch_idx})")

    def save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint = {
            "epoch": epoch,
            "batch_idx": 0,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "val_loss": val_loss,
            "config": self.model.config,
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

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
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

                print(f"\n에폭 {epoch + 1} 완료")
                print(f"  훈련 손실: {train_loss:.4f}")
                if self.val_dataloader:
                    print(f"  검증 손실: {val_loss:.4f}")
                    print(f"  Perplexity: {math.exp(val_loss):.2f}")
                print(f"  소요 시간: {epoch_time:.1f}초\n")

                self.save_checkpoint(epoch, val_loss if self.val_dataloader else train_loss)

            print("훈련 완료!")

        except KeyboardInterrupt:
            print("\n\n훈련 중단됨! 현재 상태 저장 중...")
            emergency_checkpoint = {
                "epoch": epoch,
                "batch_idx": self.global_step % len(self.train_dataloader),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "config": self.model.config,
                "is_mid_epoch": True,
            }
            path = self.checkpoint_dir / "checkpoint_latest.pt"
            torch.save(emergency_checkpoint, path)
            print(f"긴급 체크포인트 저장: {path}")
            print(f"재개하려면: python train.py --resume {path}")


def load_wikipedia_corpus(data_dir: str = "data") -> str:
    corpus_path = Path(data_dir) / "wiki_corpus.txt"

    if corpus_path.exists():
        print(f"기존 코퍼스 로드: {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            return f.read()

    print("위키피디아 코퍼스가 없습니다. 다운로드를 시작합니다...")
    print("(전체 다운로드는 시간이 오래 걸립니다. --max-articles 옵션 사용 권장)")

    from preprocessor import preprocess_wikipedia
    preprocess_wikipedia(data_dir=data_dir)

    with open(corpus_path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GPT-2 훈련")
    parser.add_argument("--batch-size", type=int, default=4, help="배치 크기")
    parser.add_argument("--seq-len", type=int, default=512, help="시퀀스 길이")
    parser.add_argument("--epochs", type=int, default=3, help="에폭 수")
    parser.add_argument("--lr", type=float, default=3e-4, help="학습률")
    parser.add_argument("--data-dir", type=str, default="data", help="데이터 디렉토리")
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["tiny", "small", "medium"],
                        help="모델 크기")
    parser.add_argument("--resume", type=str, default=None, help="체크포인트 경로")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="최대 코퍼스 문자 수 (샘플링)")
    parser.add_argument("--fp16", action="store_true",
                        help="FP16 혼합 정밀도 사용")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="N 배치마다 체크포인트 저장 (0이면 비활성화)")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    MAX_SEQ_LEN = args.seq_len
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr

    print("=" * 50)
    print("GPT-2 훈련 설정")
    print("=" * 50)
    print(f"  모델 크기: {args.model_size}")
    print(f"  배치 크기: {BATCH_SIZE}")
    print(f"  시퀀스 길이: {MAX_SEQ_LEN}")
    print(f"  에폭 수: {NUM_EPOCHS}")
    print(f"  학습률: {LEARNING_RATE}")
    print(f"  FP16: {args.fp16}")
    print(f"  체크포인트 저장 주기: {args.save_every} 배치")
    print("=" * 50)

    tokenizer = GPT2Tokenizer()

    corpus = load_wikipedia_corpus(args.data_dir)
    print(f"코퍼스 크기: {len(corpus):,} 문자 ({len(corpus) / 1e6:.1f}MB)")

    if args.max_chars and len(corpus) > args.max_chars:
        corpus = corpus[:args.max_chars]
        print(f"샘플링 후: {len(corpus):,} 문자 ({len(corpus) / 1e6:.1f}MB)")

    split_idx = int(len(corpus) * 0.95)
    train_text = corpus[:split_idx]
    val_text = corpus[split_idx:]

    print(f"훈련 데이터: {len(train_text):,} 문자")
    print(f"검증 데이터: {len(val_text):,} 문자")

    train_loader = create_dataloader(
        train_text,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_SEQ_LEN,
        stride=MAX_SEQ_LEN // 2,
        shuffle=True
    )

    val_loader = create_dataloader(
        val_text,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_SEQ_LEN,
        stride=MAX_SEQ_LEN // 2,
        shuffle=False
    )

    print(f"훈련 배치 수: {len(train_loader):,}")
    print(f"검증 배치 수: {len(val_loader):,}")

    from model import GPT2_TINY, GPT2_SMALL, GPT2_MEDIUM

    model_configs = {
        "tiny": GPT2_TINY,
        "small": GPT2_SMALL,
        "medium": GPT2_MEDIUM,
    }

    config = model_configs[args.model_size]
    config.max_seq_len = MAX_SEQ_LEN

    model = GPT(config)

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        warmup_steps=100,
        use_fp16=args.fp16,
        save_every_n_batches=args.save_every,
    )

    start_epoch = 0
    if args.resume:
        loaded_epoch, is_mid_epoch = trainer.load_checkpoint(args.resume)
        if is_mid_epoch:
            start_epoch = loaded_epoch
        else:
            start_epoch = loaded_epoch + 1

    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
