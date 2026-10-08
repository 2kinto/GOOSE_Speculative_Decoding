"""Decoder hyperparameters.

The defaults are the configuration of the paper's main results (Table 5),
fixed across models and datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GooseConfig:
    """Configuration of a decoding run.

    Attributes
    ----------
    node_budget:
        Node budget ``B`` of one draft tree, root included. A cycle confident
        enough to spend the whole budget on the chain builds one node more,
        because the root is not charged against a saturated spine.
    max_branch_depth:
        Tree depth at which branch extension stops. A branch forking off a deep
        spine node starts out beyond this depth and is simply never extended,
        so it bounds how far a branch grows, not where one may be attached.
    adjacency_top_k:
        Width ``K`` of the adjacency table.
    min_score:
        Adjacency successors scoring below this are not drafted.
    spine_branch_ratio:
        Share ``rho`` of the branch budget spent on branches hanging off the
        spine; the rest goes to branches at the root.
    max_chain:
        Longest continuation a single context match may return.
    max_extended_chain:
        Ceiling on a chain grown by repeated matching on high-confidence
        cycles. The excess over ``node_budget`` is also the budget the tree may
        borrow as confidence rises.
    long_match:
        A context match at least this long counts as high confidence.
    ema_alpha:
        Smoothing of the spine acceptance estimate that picks the spine ratio.
    ngram_lengths:
        Context lengths queried by the matcher, longest first.
    use_bigram:
        Condition adjacency lookups on the previous two tokens instead of one.
    use_context_match, use_adjacency:
        Draft sources in play. With the adjacency table alone the tree has no
        spine; with context matching alone every cycle verifies a plain chain.
    use_prefill_harvest:
        Whether the prompt's prefill logits seed the adjacency table. On is the
        paper's configuration; off leaves the branches to warm up from the
        decoding passes alone.
    topology:
        ``"spine"`` for the anisotropic tree, ``"isotropic"`` for the
        equal-budget balanced control of Section 5.3.
    branching_factor:
        Branching ``k`` of the isotropic control.
    """

    node_budget: int = 60
    max_branch_depth: int = 6
    adjacency_top_k: int = 10
    min_score: float = 0.01
    spine_branch_ratio: float = 0.5
    max_chain: int = 20
    max_extended_chain: int = 100
    long_match: int = 8
    ema_alpha: float = 0.3
    ngram_lengths: Tuple[int, ...] = (5, 4, 3)
    spine_ratio_tiers: Tuple[float, float, float] = (0.15, 0.30, 0.50)

    use_bigram: bool = True
    use_context_match: bool = True
    use_adjacency: bool = True
    # Seed the adjacency table from the prompt's own logits during prefill
    # (Appendix C.1). Off, the table starts empty and fills only from the
    # passes the decoding loop runs.
    use_prefill_harvest: bool = True
    # The confidence signal of Section 4.3 and what it drives: chain
    # extension, the budget extension and the confidence-driven shift of the
    # spine ratio (the acceptance-rate tiers stay active). Off is the
    # configuration of Table 2's "w/o consensus bypass" row: every cycle
    # builds the tree at the base budget.
    use_confidence: bool = True

    topology: str = "spine"
    branching_factor: int = 3

    def __post_init__(self) -> None:
        if self.topology not in ("spine", "isotropic"):
            raise ValueError(f"unknown topology: {self.topology!r}")
        if not (self.use_context_match or self.use_adjacency):
            raise ValueError("at least one draft source must be enabled")
        if self.node_budget < 2:
            raise ValueError("node_budget must leave room for the root and one draft")
        if self.max_branch_depth < 1 or self.adjacency_top_k < 1:
            raise ValueError("max_branch_depth and adjacency_top_k must be positive")
        if self.branching_factor < 1:
            raise ValueError("branching_factor must be positive")
        if self.long_match < 1:
            raise ValueError("long_match is a chain length")
        if not 0.0 <= self.min_score < 1.0:
            # Scores are renormalised probabilities, so a threshold of 1 would
            # prune every candidate and leave the decoder guessing one token a
            # forward with no error to show for it.
            raise ValueError("min_score must be a probability below 1")
        if not 0.0 <= self.spine_branch_ratio <= 1.0:
            raise ValueError("spine_branch_ratio is a share of the branch budget")
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha is a smoothing weight")
        if not self.ngram_lengths or min(self.ngram_lengths) < 1:
            raise ValueError("ngram_lengths must contain positive lengths")
        if self.max_chain < 1 or self.max_extended_chain < self.max_chain:
            raise ValueError("max_extended_chain cannot be shorter than max_chain")
        if len(self.spine_ratio_tiers) != 3 or not all(
                0.0 <= tier <= 1.0 for tier in self.spine_ratio_tiers):
            raise ValueError("spine_ratio_tiers is three shares in [0, 1]")

    @property
    def budget_extension(self) -> int:
        """Extra nodes a fully confident cycle may spend on top of ``node_budget``."""
        return max(0, self.max_extended_chain - self.node_budget)

    def spine_ratio(self, spine_accept_rate: float) -> float:
        """Share of the budget reserved for the spine, from the acceptance EMA.

        Cycles whose spine has been rejected early recently give their budget
        to the branches instead (Section 4.3).
        """
        low, medium, high = self.spine_ratio_tiers
        if spine_accept_rate < 0.2:
            return low
        if spine_accept_rate < 0.4:
            return medium
        return high
