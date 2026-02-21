import tiktoken


class GPT2Tokenizer:

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.eos_token = "<|endoftext|>"
        self.eos_token_id = self.tokenizer.encode(
            self.eos_token,
            allowed_special={self.eos_token}
        )[0]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if add_special_tokens:
            allowed = {self.eos_token}
        else:
            allowed = set()

        token_ids = self.tokenizer.encode(text, allowed_special=allowed)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer()

    text = "Hello, world! This is a test."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"원본 텍스트: {text}")
    print(f"토큰 ID: {tokens}")
    print(f"디코딩: {decoded}")
    print(f"어휘 크기: {tokenizer.vocab_size}")
