"""MHA + RoPE vs FA + RoPE."""

from cohebot.attention.flash import FlashAttention
from cohebot.attention.mha import MultiHeadAttention

from . import make_data, print_table, run


def main():
    data = make_data()
    results = {
        "MHA + RoPE": run(MultiHeadAttention, data),
        "FA + RoPE": run(FlashAttention, data),
    }
    print_table(results)


if __name__ == "__main__":
    main()
