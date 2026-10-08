"""Counters collected during a decoding run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecodeStats:
    """Per-request statistics.

    ``compression_ratio`` is the paper's tau: committed tokens per target-model
    forward pass, prefill included.
    """

    generated_tokens: int = 0
    forward_calls: int = 0
    # Tokens a cycle committed past max_new_tokens and that were cut from the
    # output. The per-source counters below describe what the decoder did, so
    # they still include these; this field is what reconciles the two.
    truncated_tokens: int = 0
    # Accepted tokens discarded because a stop token preceded them in the same
    # cycle. Like truncated_tokens, the per-source counters still include them.
    eos_dropped_tokens: int = 0
    wall_time: float = 0.0
    model_time: float = 0.0
    prefill_time: float = 0.0

    # Cycles by the route taken (Appendix B.2).
    spine_tree_cycles: int = 0
    branch_tree_cycles: int = 0
    chain_cycles: int = 0
    ar_cycles: int = 0

    # Draft nodes offered, excluding the anchor the tree is rooted at, so that
    # draft_nodes == spine_nodes + branch_nodes on every route.
    draft_nodes: int = 0
    spine_nodes: int = 0
    branch_nodes: int = 0

    # Draft nodes accepted, by source.
    spine_accepted: int = 0
    branch_accepted: int = 0
    bonus_tokens: int = 0

    # Cycles whose accepted path left the spine for a branch, and the tokens
    # those cycles won past the spine break (Section 4.2).
    spine_continuations: int = 0
    spine_continuation_tokens: int = 0

    @property
    def committed_tokens(self) -> int:
        """Every token a walk accepted or the model produced itself.

        Counted before the length limit and a stop token cut the output back.
        """
        return (self.generated_tokens + self.truncated_tokens
                + self.eos_dropped_tokens)

    @property
    def compression_ratio(self) -> float:
        return self.generated_tokens / max(1, self.forward_calls)

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.wall_time if self.wall_time > 0 else 0.0
