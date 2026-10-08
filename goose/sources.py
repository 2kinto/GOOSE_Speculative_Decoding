"""The two draft sources, as interfaces.

Goose assembles each cycle's candidate pool from two sources: a *spine source*
proposes one chain of tokens the target model is likely to accept in sequence,
and a *branch source* proposes ranked alternatives at any node. The paper's
sources are context matching and the bigram adjacency table, and they are the
defaults. Anything that speaks the same two protocols can stand in for either,
and the tree construction, verification and walk stay exactly as they are.

Both protocols are called on the decoder's own schedule and see the same
observations, so a source needs no knowledge of the loop:

    spine.on_prefill(prompt_tokens)      once, before decoding
    spine.on_commit(committed_tokens)    after every cycle
    spine.propose(tokens)                the chain to speculate from here
    spine.confidence(tokens, chain)      how far to trust it (Section 4.3)

    branches.observe(token_ids, logits, prev_tokens)   after every forward
    branches.successors(token, prev_token, limit)      ranked alternatives
    branches.has_successors(token, prev_token)          cheap emptiness probe
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch


@runtime_checkable
class SpineSource(Protocol):
    """Proposes the chain of a cycle and says how far to trust it."""

    def on_prefill(self, tokens: Sequence[int]) -> None:
        """The prompt, once per ``decode`` call, before any ``propose``.

        A source reused across calls sees this once per request; treat it as
        the start of a request, not as more text.
        """

    def on_commit(self, tokens: Sequence[int]) -> None:
        """Tokens the model has just committed, in order.

        Called after every cycle that continues generation, so a source may
        assume that the concatenation of the prompt and every ``on_commit``
        batch is the sequence ``propose`` is queried on. It is not called for
        the cycle that stops on an end-of-sequence token.
        """

    def propose(self, tokens: Sequence[int]) -> List[int]:
        """The chain to speculate from the end of ``tokens``.

        ``tokens`` is the prompt plus everything the model has committed, and,
        during a confident cycle's extension, plus tokens already drafted this
        cycle; the last element is the position to continue from. Return
        ``[]`` when there is nothing to propose. The decoder uses the chain
        as returned, so cap its length here.
        """

    def confidence(self, tokens: Sequence[int], chain: Sequence[int]) -> float:
        """Confidence in ``chain`` proposed from the end of ``tokens``, in [0, 1].

        The decoder scales the node budget and the spine's share of it with
        this value, grows the chain by further ``propose`` calls when it is
        above 0.6, and, at exactly 1.0, keeps growing only while each further
        proposal also scores 1.0 (Section 4.3). Zero is "no evidence beyond
        the chain existing".
        """


@runtime_checkable
class BranchSource(Protocol):
    """Ranked alternatives at any node, learned from the model's own logits."""

    def observe(self, token_ids: Sequence[int], logits: torch.Tensor,
                prev_tokens: Optional[Sequence[Optional[int]]] = None) -> int:
        """Every forward pass the decoder runs: prefill, tree, chain and step.

        ``logits`` is ``[n_positions, vocab]``, ``token_ids[i]`` is the token
        that produced row ``i`` and ``prev_tokens[i]`` the one before it (for a
        tree node, its parent; ``None`` when unknown or when the decoder runs
        without bigram context). Rows of rejected draft nodes are included.
        Return how many rows were used.
        """

    def successors(self, token_id: int, prev_token: Optional[int] = None,
                   limit: Optional[int] = None) -> List[Tuple[int, float]]:
        """``(token, score)`` pairs, best first, at most ``limit``.

        Scores are probabilities in [0, 1]; the tree builder prunes below
        ``GooseConfig.min_score`` and sizes branches by them. The decoder calls
        this with ``prev_token`` and ``limit`` as keywords.
        """

    def has_successors(self, token_id: int, prev_token: Optional[int] = None) -> bool:
        """Whether ``successors`` would return anything, without building it."""
