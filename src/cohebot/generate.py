import argparse
import torch

from .model import GPT, GPTConfig, GPT2_TINY
from .tokenizer import GPT2Tokenizer


def load_model(checkpoint_path: str, device: str = "auto") -> tuple[GPT, GPTConfig]:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def generate_text(
    model: GPT,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "cpu"
) -> str:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    generated_text = tokenizer.decode(output_ids[0].tolist())

    return generated_text


def interactive_mode(model: GPT, tokenizer: GPT2Tokenizer, device: str):
    print("\n=== GPT 텍스트 생성기 ===")
    print("'quit' 또는 'exit'를 입력하면 종료합니다.\n")

    while True:
        prompt = input("프롬프트: ").strip()

        if prompt.lower() in ("quit", "exit", "q"):
            print("종료합니다.")
            break

        if not prompt:
            continue

        print("\n생성 중...\n")

        print("--- Greedy (temperature=0.1) ---")
        text = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=50,
            temperature=0.1,
            device=device
        )
        print(text)

        print("\n--- Top-K=50, temperature=0.8 ---")
        text = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            device=device
        )
        print(text)

        print("\n--- Top-P=0.9, temperature=1.0 ---")
        text = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=50,
            temperature=1.0,
            top_p=0.9,
            device=device
        )
        print(text)
        print("\n" + "=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="GPT 텍스트 생성")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="모델 체크포인트 경로"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="생성 시작 프롬프트 (없으면 대화형 모드)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="생성할 최대 토큰 수"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="샘플링 온도"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-K 샘플링"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-P 샘플링"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="디바이스 (auto, cpu, cuda, mps)"
    )
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"디바이스: {device}")

    tokenizer = GPT2Tokenizer()

    try:
        model, config = load_model(args.checkpoint, device)
        print(f"모델 로드 완료: {args.checkpoint}")
    except FileNotFoundError:
        print(f"체크포인트를 찾을 수 없습니다: {args.checkpoint}")
        print("테스트용 랜덤 초기화 모델을 사용합니다.")
        config = GPT2_TINY
        model = GPT(config).to(device)

    if args.prompt:
        text = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )
        print(f"\n생성된 텍스트:\n{text}")
    else:
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
