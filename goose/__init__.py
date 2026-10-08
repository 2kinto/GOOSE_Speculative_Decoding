"""Goose: anisotropic speculation trees for training-free speculative decoding.

Reference implementation for the COLM 2026 paper
"Goose: Anisotropic Speculation Trees for Training-Free Speculative Decoding".

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from goose import GooseConfig, decode

    tokens, stats = decode(model, tokenizer, input_ids, max_new_tokens=512)
    print(stats.compression_ratio, stats.tokens_per_second)
"""

from .adjacency import AdjacencyTable
from .config import GooseConfig
from .context_match import ContextMatcher, match_confidence
from .decoding import decode, decode_autoregressive, stop_tokens
from .sources import BranchSource, SpineSource
from .stats import DecodeStats
from .tree import SpeculationTree, build_isotropic_tree, build_spine_tree
from .verify import greedy_walk, verify_tree

__version__ = "1.0.0"

__all__ = [
    "AdjacencyTable",
    "BranchSource",
    "ContextMatcher",
    "DecodeStats",
    "GooseConfig",
    "SpeculationTree",
    "SpineSource",
    "build_isotropic_tree",
    "build_spine_tree",
    "decode",
    "decode_autoregressive",
    "greedy_walk",
    "match_confidence",
    "stop_tokens",
    "verify_tree",
]
