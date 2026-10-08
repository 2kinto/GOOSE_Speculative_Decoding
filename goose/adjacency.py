"""Adjacency table: the transition draft source.

The table is Token Recycling's adjacency matrix (Luo et al., ACL 2025), which
keeps, for every vocabulary entry, the tokens most likely to follow it and
refreshes them from the logits the target model produces anyway. What is added
here is the second tier: a lookup may be conditioned on the previous two tokens
rather than one (Section 4.1, Appendix C.1).

Every forward pass the target model runs already contains a next-token
distribution for every position it processed, the prefill and each verification
alike, including the positions of rejected branches. The table keeps the top-K
of each of them and serves them back as branch candidates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


class AdjacencyTable:
    """Two-tier top-K successor table: the paper's branch source.

    The unigram tier lives in two ``[vocab_size, top_k]`` GPU tensors, so a
    lookup is a slice and needs no host round-trip. The bigram tier is a hash
    table keyed on the two most recent tokens and is consulted first: two
    tokens of context narrow the prediction (the successors of ``f`` after
    ``def`` are fewer than the successors of ``f`` alone).
    """

    def __init__(self, vocab_size: int, top_k: int = 10, device=None) -> None:
        self.vocab_size = vocab_size
        self.top_k = top_k
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.unigram_ids = torch.full(
            (vocab_size, top_k), -1, dtype=torch.long, device=self.device)
        self.unigram_scores = torch.zeros(
            (vocab_size, top_k), dtype=torch.float16, device=self.device)
        self.observed = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)

        self.bigram: Dict[Tuple[int, int], Tuple[List[int], List[float]]] = {}
        self._lookups = 0
        self._bigram_hits = 0

    def harvest(
        self,
        token_ids: Sequence[int],
        logits: torch.Tensor,
        prev_tokens: Optional[Sequence[Optional[int]]] = None,
    ) -> int:
        """Record the top-K successors of every position of one forward pass.

        ``logits`` is ``[n_positions, vocab_size]`` and ``token_ids[i]`` is the
        token that produced row ``i``. ``prev_tokens[i]`` is the token before
        it; positions with no predecessor update the unigram tier only.
        """
        n = min(len(token_ids), logits.shape[0])
        if n == 0:
            return 0

        values, ids = torch.topk(logits[:n], self.top_k, dim=-1)
        scores = torch.softmax(values, dim=-1)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

        # One pass often scores the same token at several positions: a draft
        # tree repeats tokens, and so does any real prompt. Assigning with
        # repeated row indices resolves the collision in an unspecified way,
        # and separately for each tensor, which can leave an entry holding the
        # successors of one position and the scores of another. Keep the last
        # position of each token, the freshest statistics for it.
        positions = sorted({token: i for i, token in enumerate(token_ids[:n])}.values())
        keep = torch.as_tensor(positions, dtype=torch.long, device=ids.device)
        rows = torch.as_tensor([token_ids[i] for i in positions],
                               dtype=torch.long, device=self.device)

        self.unigram_ids[rows] = ids.index_select(0, keep).to(self.device)
        self.unigram_scores[rows] = scores.index_select(0, keep).to(
            self.device, torch.float16)
        self.observed[rows] = True

        if prev_tokens is not None:
            id_rows = ids.tolist()
            score_rows = scores.tolist()
            for i, previous in enumerate(prev_tokens[:n]):
                if previous is not None:
                    self.bigram[(previous, token_ids[i])] = (id_rows[i], score_rows[i])
        return n

    # ``observe`` is the BranchSource name for the harvest.
    def observe(self, token_ids: Sequence[int], logits: torch.Tensor,
                prev_tokens: Optional[Sequence[Optional[int]]] = None) -> int:
        return self.harvest(token_ids, logits, prev_tokens)

    def successors(
        self,
        token_id: int,
        prev_token: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Successors of ``token_id`` with their scores, most likely first."""
        k = limit or self.top_k
        self._lookups += 1

        if prev_token is not None:
            entry = self.bigram.get((prev_token, token_id))
            if entry is not None:
                self._bigram_hits += 1
                return list(zip(entry[0][:k], entry[1][:k]))

        if not self.observed[token_id]:
            return []          # a shortcut: the row is all padding and would filter to nothing
        ids = self.unigram_ids[token_id, :k].tolist()
        scores = self.unigram_scores[token_id, :k].float().tolist()
        return [(i, s) for i, s in zip(ids, scores) if i >= 0]

    def has_successors(self, token_id: int, prev_token: Optional[int] = None) -> bool:
        """Whether a lookup would return anything, without building the list.

        The router asks this once a cycle before the builder asks for the list
        itself, so it neither materialises the transfer nor counts towards the
        bigram hit rate.
        """
        if prev_token is not None and (prev_token, token_id) in self.bigram:
            return True
        return bool(self.observed[token_id])

    @property
    def n_unigram_entries(self) -> int:
        return int(self.observed.sum())

    @property
    def n_bigram_entries(self) -> int:
        return len(self.bigram)

    @property
    def bigram_hit_rate(self) -> float:
        return self._bigram_hits / self._lookups if self._lookups else 0.0
