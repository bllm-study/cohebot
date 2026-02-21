import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import GPT2Tokenizer


class TextDataset(Dataset):

    def __init__(
        self,
        text: str,
        tokenizer: GPT2Tokenizer,
        max_length: int = 256,
        stride: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        self.token_ids = tokenizer.encode(text)

        self.sequences = []
        for i in range(0, len(self.token_ids) - max_length, stride):
            input_seq = self.token_ids[i:i + max_length]
            target_seq = self.token_ids[i + 1:i + max_length + 1]
            self.sequences.append((input_seq, target_seq))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq, target_seq = self.sequences[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )


def create_dataloader(
    text: str,
    tokenizer: GPT2Tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    dataset = TextDataset(text, tokenizer, max_length, stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer()

    sample_text = """
    인공지능(AI)은 인간의 학습, 추론, 지각 등의 지능적 행동을
    컴퓨터가 모방할 수 있도록 하는 기술입니다. 머신러닝은 AI의
    한 분야로, 데이터로부터 패턴을 학습합니다.
    """ * 100

    dataloader = create_dataloader(
        text=sample_text,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=64,
        stride=32
    )

    for batch_idx, (x, y) in enumerate(dataloader):
        print(f"배치 {batch_idx}:")
        print(f"  입력 shape: {x.shape}")
        print(f"  타겟 shape: {y.shape}")
        print(f"  입력 예시: {tokenizer.decode(x[0].tolist()[:20])}...")
        if batch_idx >= 2:
            break
