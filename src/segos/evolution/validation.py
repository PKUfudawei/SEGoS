from functools import lru_cache
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe

@lru_cache(maxsize=1)
def tokenizer():
    mergeable_ranks = load_tiktoken_bpe(
        str(Path(__file__).with_name("assets") / "cl100k_base.tiktoken"),
        expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    )
    special_tokens = {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        "<|fim_middle|>": 100259,
        "<|fim_suffix|>": 100260,
        "<|endofprompt|>": 100276,
    }
    return tiktoken.Encoding(**{
        "name": "cl100k_base",
        "pat_str": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    })

def within_edit_budget(original, candidate, budget=50):
    """Bound net added cl100k_base tokens, independent of the execution model."""
    enc = tokenizer()
    return len(enc.encode(candidate, disallowed_special=())) - len(enc.encode(original, disallowed_special=())) <= budget
