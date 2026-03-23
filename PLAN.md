# GPU Cloud 학습 스크립트 계획

## 목표

RunPod / Hyperbolic 등 GPU Cloud 인스턴스에서 **명령어 하나로** 환경 세팅부터 학습까지 끝나는 올인원 학습 스크립트.

---

## 현재 상태

- `cohebot-train` CLI 존재, TOML 설정 파일 지원 (`--config` 필수)
- attention 모듈(RoPE, MHA, GQA, FlashAttention)은 placeholder — 구현 진행 중
- 데이터: HuggingFace Hub (`bllm-study/kowiki-corpus`)에서 전처리된 corpus 다운로드
- 모델 설정: 모두 TOML 파일로 관리 (`CoheLLMBotConfig`)
- W&B 로깅, HF Hub 체크포인트 업로드, Multi-GPU DDP 지원

## 타겟 환경

| 항목 | 값 |
|------|-----|
| 클라우드 | RunPod (1순위), Hyperbolic (대안) |
| Python | 3.10+ |
| 패키지 매니저 | uv |

---

## 파일 구조

```
configs/
├── default.toml    # 로컬 개발용
└── cloud.toml      # GPU Cloud용

scripts/
├── setup_cloud.sh  # 환경 세팅 (uv, clone, 의존성)
├── train.sh        # 학습 실행 래퍼 (Single/Multi-GPU)
└── run.sh          # 올인원 (setup + train)
```

---

## 사용법

### 원커맨드 (클라우드 인스턴스)

```bash
curl -sSL https://raw.githubusercontent.com/bllm-study/cohebot/main/scripts/run.sh | bash
```

### Multi-GPU

```bash
NUM_GPUS=4 bash scripts/train.sh --config configs/cloud.toml
```

### W&B 로깅

TOML `[logging]` 섹션에서 활성화:

```toml
[logging]
wandb = true
wandb_project = "cohebot"
wandb_run_name = "run-1"
```

### 체크포인트 HF Hub 자동 업로드

TOML `[checkpoint]` 섹션에서 설정:

```toml
[checkpoint]
dir = "/workspace/checkpoints"
upload_repo = "bllm-study/cohebot-ckpt"
```

### 로컬 개발

```bash
uv run cohebot-train --config configs/default.toml
```

---

## 남은 작업

- [x] 데이터 소스 확정 → HF Hub
- [x] W&B 로깅 연동
- [x] 체크포인트 외부 백업 (HF Hub)
- [x] Multi-GPU DDP 지원
- [ ] HF Hub에 전처리된 corpus 실제 업로드
- [ ] RunPod에서 dry-run 테스트
- [ ] CLI 분리 리팩토링 후 스크립트 업데이트
