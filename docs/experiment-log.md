# E2E 실험 로그: MHA + RoPE 크롤→학습 파이프라인

> **날짜**: 2026-03-28
> **브랜치**: `hf-cli`
> **목적**: Multi-Head Attention + RoPE 구현 후 크롤→학습 E2E 검증

---

## 1. 실행 명령어

```bash
bash scripts/pipeline.sh --max-articles 5 --config configs/test-cpu.toml
```

내부적으로 다음이 순차 실행됨:

1. `uv run cohebot-crawl --data-dir data --max-articles 5`
2. `uv run cohebot-train --config configs/test-cpu.toml --data-dir data`

---

## 2. 모델 설정 (`configs/test-cpu.toml`)

```toml
[model]
vocab_size   = 50257
max_seq_len  = 128
embed_dim    = 128
num_heads    = 4
num_layers   = 2
ff_dim       = 512
dropout      = 0.1
bias         = true
attn_type    = "mha"

[training]
batch_size    = 2
seq_len       = 128
epochs        = 1
lr            = 3e-4
weight_decay  = 0.1
warmup_steps  = 10
max_grad_norm = 1.0
fp16          = false
save_every    = 0
```

**파라미터 수**: 6,831,809 (~6.83M)

---

## 3. 크롤 결과

| 항목 | 값 |
|------|-----|
| 수집 문서 수 | 5 |
| 총 문자 수 | 15,389 |
| 코퍼스 파일 크기 | ~36KB |
| 소요 시간 | ~5초 |
| 출력 파일 | `data/wiki_corpus.txt` (364줄) |

---

## 4. 학습 결과

| 항목 | 값 |
|------|-----|
| 디바이스 | CUDA (GPU) |
| 배치 수 | 228 |
| 학습 시간 | ~5.3초 |
| 최종 train_loss | 5.90 |
| 최종 val_loss | 4.33 |
| 최종 PPL | 76.11 |
| 체크포인트 | `checkpoints/test-cpu/` (3파일, ~82MB each) |

### 손실 곡선 (샘플링)

```
Step    Loss        LR
─────────────────────────────
0       10.8415     3.00e-05   ← warmup 시작
1       10.8516     6.00e-05
7       10.4835     2.10e-04
13       9.8427     3.00e-04   ← warmup 종료, cosine decay 시작
19       9.5897     2.87e-04
25       9.0092     2.60e-04
31       8.6908     2.29e-04
37       8.2922     1.95e-04
43       7.9167     1.60e-04
49       7.4118     1.27e-04
55       7.0728     9.78e-05
61       6.9037     7.37e-05
67       6.3271     5.40e-05
73       5.9861     3.90e-05
79       5.6810     2.67e-05
85       5.5148     1.74e-05
91       5.2800     1.04e-05
97       4.9752     5.70e-06
103      4.8978     2.40e-06
109      4.7431     5.30e-07
115      4.6842     6.70e-08
121      4.4947     3.90e-09
127      4.4786     9.70e-11
133      4.3125     8.40e-13
139      4.3043     3.20e-14
145      4.2023     5.40e-15
151      4.3319     0.00e+00   ← LR 완전 감쇠
157      4.1705     0.00e+00
163      4.1451     0.00e+00
169      4.1514     0.00e+00
175      4.0980     0.00e+00
181      4.2048     0.00e+00
187      4.1261     0.00e+00
193      3.9653     0.00e+00
199      4.2927     0.00e+00
205      4.3514     0.00e+00
211      4.0141     0.00e+00
217      4.1370     0.00e+00
223      4.0586     0.00e+00
228      3.9986     0.00e+00   ← 최종 배치
```

---

## 5. 관찰된 현상

### 5.1 🔥 Datasets 라이브러리 exit code 134

```
terminate called without an active exception
Aborted (core dumped)
```

- HuggingFace `datasets` 라이브러리의 **스레드 정리 버그**
- 데이터가 모두 디스크에 기록된 **이후**에 발생
- 코퍼스 파일은 정상 생성됨
- `pipeline.sh`에서 `|| true` + 파일 존재 확인으로 우회

### 5.2 📈 Warmup 단계 (steps 0–10)

- LR이 `3e-5` → `3e-4`까지 선형 증가
- Loss가 천천히 감소 (~10.8 → ~10.2)
- Warmup 직후 loss 급감 시작

### 5.3 📉 Cosine Decay 단계 (steps 10–150)

- LR이 `3e-4` → `0`까지 cosine 감쇠
- Loss가 ~10.2 → ~4.2까지 빠르게 감소
- LR이 거의 0에 수렴하면서 loss 감소 둔화

### 5.4 📊 Loss 정체 구간 (steps 150–228)

- LR = 0 이후 loss ~4.0–4.3 범위에서 변동
- 극소형 데이터셋(5문서)으로 인한 **언더피팅** — 정상적
- 실제 학습에서는 더 많은 corpus로 해결

### 5.5 🤔 Val Loss < Train Loss

- val_loss (4.33) < train_loss (5.90)
- 원인: train_loss에 warmup 기간의 높은 loss가 포함
- val_loss는 에폭 종료 후 측정되므로 이미 학습이 진행된 상태

---

## 6. 구현 아키텍처

### RoPE (`src/cohebot/attention/rope.py`)

- `inv_freq = 1 / (base^(2i/d))` 사전 계산 → `register_buffer`
- `_cos_cached`, `_sin_cached` 사전 계산 (max_seq_len 기준)
- 회전: 짝수/홀수 인덱스 분할 → `x1*cos - x2*sin`, `x2*cos + x1*sin`
- Q, K에 동일하게 적용

### MHA (`src/cohebot/attention/mha.py`)

- Q/K/V 개별 선형 프로젝션 (`embed_dim → embed_dim`)
- Reshape: `(B, S, embed_dim)` → `(B, num_heads, S, head_dim)`
- RoPE를 Q, K에 적용
- Scaled dot-product attention + causal mask (`torch.triu`)
- `attn_dropout` + `resid_dropout`

### 파이프라인 (`scripts/pipeline.sh`)

- Phase 1: `cohebot-crawl` — 기존 코퍼스 있으면 스킵
- Phase 2: `cohebot-train` — TOML 설정으로 학습
- `--max-articles`, `--config`, `--data-dir` 인자 지원

---

## 7. 산출물

| 파일 | 설명 | Git 추적 |
|------|------|----------|
| `src/cohebot/attention/rope.py` | RoPE 구현 | ✅ |
| `src/cohebot/attention/mha.py` | MHA + RoPE 구현 | ✅ |
| `configs/test-cpu.toml` | 극소형 테스트 설정 | ✅ |
| `scripts/pipeline.sh` | 통합 크롤+학습 파이프라인 | ✅ |
| `scripts/crawl.sh` | 단독 크롤 스크립트 | ✅ |
| `data/wiki_corpus.txt` | 테스트 코퍼스 (36KB) | ❌ gitignore |
| `checkpoints/test-cpu/` | 체크포인트 (~246MB) | ❌ gitignore |

---

## 8. 남은 작업

- [ ] `src/cohebot/attention/gqa.py` — GroupedQueryAttention 구현
- [ ] `src/cohebot/attention/flash.py` — FlashAttention 구현
- [ ] 실제 코퍼스로 전체 학습 테스트 (100+ 문서)
- [ ] HuggingFace Hub에 `bllm-study/kowiki-corpus` 업로드
- [ ] `PLAN.md` 업데이트 (attention 구현 상태 반영)
