"""Context matching: the spine draft source.

The pool is the prompt plus everything committed so far. Given the current
suffix, the matcher returns the continuation that followed the same n-gram
earlier in the pool: the classic prompt-lookup draft (Saxena, 2023), queried
at several n-gram lengths, longest first. The same index yields the consensus
signal used by the confidence schedule (Section 4.3) at no extra cost.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


class ContextMatcher:
    """Positional n-gram index over prompt and generated tokens.

    This is the paper's spine source. It keeps its own copy of the token
    stream, so ``propose`` and ``confidence`` only need the tail of the
    decoder's sequence.
    """

    def __init__(self, ngram_lengths: Sequence[int] = (5, 4, 3),
                 max_continuation: int = 20, long_match: int = 8) -> None:
        self.ngram_lengths = tuple(sorted(ngram_lengths, reverse=True))
        self.max_continuation = max_continuation
        self.long_match = long_match
        self._tokens: List[int] = []
        self._index: Dict[int, Dict[Tuple[int, ...], List[int]]] = {
            n: defaultdict(list) for n in self.ngram_lengths
        }

    @property
    def context_window(self) -> int:
        """Number of trailing tokens a query needs to exercise every length."""
        return self.ngram_lengths[0]

    # the SpineSource protocol

    def on_prefill(self, tokens: Sequence[int]) -> None:
        self.extend(tokens)

    def on_commit(self, tokens: Sequence[int]) -> None:
        self.extend(tokens)

    def propose(self, tokens: Sequence[int]) -> List[int]:
        return self.match(tokens[-self.context_window:])

    def confidence(self, tokens: Sequence[int], chain: Sequence[int]) -> float:
        """Confidence in a context match, in [0, 1] (Section 4.3).

        Two indicators drive it. Consensus, every earlier occurrence of the
        current context (at least two) continuing with the same token, is the
        strong one and
        saturates the scale. Failing that, a long match is itself evidence:
        chains at or past ``long_match`` tokens start at 0.7 and approach 0.9,
        shorter ones scale linearly up to 0.6.
        """
        return match_confidence(len(chain), self.consensus(tokens[-self.context_window:]),
                                self.long_match)

    # the index itself

    def extend(self, tokens: Sequence[int]) -> None:
        """Append newly committed tokens to the pool and index them."""
        if not tokens:
            return
        start = len(self._tokens)
        self._tokens.extend(tokens)
        for n in self.ngram_lengths:
            index = self._index[n]
            # Only occurrences with a token after them are indexed. The n-gram
            # sitting at the very end of the stream is the query itself: it has
            # nothing to copy, and counting it as a hit would keep the shorter
            # n-grams from ever being consulted.
            for i in range(max(0, start - n), len(self._tokens) - n):
                index[tuple(self._tokens[i:i + n])].append(i)

    def match(self, suffix: Sequence[int]) -> List[int]:
        """Continuation of the longest n-gram of ``suffix`` seen before.

        Occurrences are indexed in order, so the first one carries the longest
        continuation; ``[]`` means no source proposed anything this cycle.
        """
        for n in self.ngram_lengths:
            if len(suffix) < n:
                continue
            positions = self._index[n].get(tuple(suffix[-n:]))
            if not positions:
                continue
            start = positions[0] + n
            return self._tokens[start:start + self.max_continuation]
        return []

    def consensus(self, suffix: Sequence[int]) -> bool:
        """Whether every earlier occurrence agrees on the next token.

        Only the longest n-gram length with an earlier occurrence is consulted,
        and it needs at least two occurrences: a single occurrence agrees with
        itself, which says nothing, while two or more that all continue
        identically are the signal that the match is template rather than
        content.
        """
        for n in self.ngram_lengths:
            if len(suffix) < n:
                continue
            positions = self._index[n].get(tuple(suffix[-n:]))
            if not positions:
                continue
            starts = {p + n for p in positions}
            if len(starts) < 2:
                return False
            return len({self._tokens[p] for p in starts}) == 1
        return False


def match_confidence(chain_length: int, consensus: bool, long_match: int = 8) -> float:
    """The confidence schedule of Section 4.3, as a pure function of its two indicators."""
    if chain_length == 0:
        return 0.0
    if consensus:
        return 1.0
    if chain_length >= long_match:
        return min(0.9, 0.7 + 0.2 * (chain_length - long_match) / 12.0)
    return 0.6 * chain_length / long_match
