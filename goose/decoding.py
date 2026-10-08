"""The Goose decoding loop (Section 4, Algorithm 2).

Each cycle draws candidates from the two training-free sources, packs them into
one tree, verifies them in a single forward pass and commits the longest
accepted path. A draft token is committed only where it equals the token the
target model would have produced anyway, so the output is that of greedy
autoregressive decoding, up to floating-point near-ties between the top two
logits.
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
from transformers import DynamicCache

from .adjacency import AdjacencyTable
from .config import GooseConfig
from .context_match import ContextMatcher
from .runtime import crop_cache, timed_forward
from .sources import BranchSource, SpineSource
from .stats import DecodeStats
from .tree import SPINE, SpeculationTree, build_isotropic_tree, build_spine_tree
from .verify import verify_tree

# Above this confidence the context match is trusted enough to be grown by
# further lookups before the tree is built (Section 4.3).
CHAIN_EXTENSION_CONFIDENCE = 0.6


def stop_tokens(model, tokenizer,
                eos_token_id: Optional[Union[int, Iterable[int]]] = None) -> frozenset:
    """Every id at which the model considers the sequence finished.

    Current chat checkpoints usually list more than one: Llama-3-8B-Instruct
    stops on ``<|eot_id|>`` and on ``<|end_of_text|>``, Qwen3 on ``<|im_end|>``
    and on ``<|endoftext|>``, while ``tokenizer.eos_token_id`` names only the
    first of each pair. Honouring just that one lets a generation continue
    after the model has finished.
    """
    if eos_token_id is None:
        generation = getattr(model, "generation_config", None)
        eos_token_id = getattr(generation, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    return frozenset(token for token in (eos_token_id or []) if token is not None)


@torch.no_grad()
def decode(
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 512,
    eos_token_id: Optional[Union[int, Iterable[int]]] = None,
    config: Optional[GooseConfig] = None,
    spine_source: Optional[SpineSource] = None,
    branch_source: Optional[BranchSource] = None,
) -> Tuple[torch.LongTensor, DecodeStats]:
    """Generate a continuation of ``input_ids`` (shape ``[1, prompt_length]``).

    Returns the prompt followed by the generated tokens, and the run's counters.

    The two draft sources default to the paper's: context matching over the
    prompt and generated text for the spine, the bigram adjacency table for the
    branches. Pass objects speaking the protocols in ``goose.sources`` to swap
    either one; the tree, the mask and the walk do not change. A source passed
    here is used as-is, so a spine source that carries state across requests
    keeps it.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("Goose decodes one sequence at a time, shaped [1, prompt_length]")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    _require_maskable_attention(model)
    stops = stop_tokens(model, tokenizer, eos_token_id)
    config = config or GooseConfig()

    # A source passed explicitly and a config flag that switches that source
    # off contradict each other; refuse rather than pick one quietly.
    if spine_source is not None and not config.use_context_match:
        raise ValueError("spine_source was given but config.use_context_match is False")
    if branch_source is not None and not config.use_adjacency:
        raise ValueError("branch_source was given but config.use_adjacency is False")

    # The clock starts before the draft sources are allocated, so that the
    # setup speculation needs is charged to speculation. Autoregressive
    # decoding has none, and timing only the loop would quietly hand this
    # method a head start.
    started = time.perf_counter()
    if spine_source is None and config.use_context_match:
        spine_source = ContextMatcher(config.ngram_lengths, config.max_chain,
                                      config.long_match)
    if branch_source is None and config.use_adjacency:
        branch_source = AdjacencyTable(model.config.vocab_size, config.adjacency_top_k,
                                       input_ids.device)
    session = _Session(model, input_ids, stops, config, spine_source, branch_source)
    return session.run(max_new_tokens, started)


@torch.no_grad()
def decode_autoregressive(
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 512,
    eos_token_id: Optional[Union[int, Iterable[int]]] = None,
) -> Tuple[torch.LongTensor, DecodeStats]:
    """Greedy decoding with a KV cache: the reference output and the speed baseline."""
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("one sequence at a time, shaped [1, prompt_length]")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    stops = stop_tokens(model, tokenizer, eos_token_id)

    stats = DecodeStats()
    generated = input_ids
    current = input_ids
    past = DynamicCache()

    started = time.perf_counter()
    for step in range(max_new_tokens):
        output = timed_forward(model, stats, input_ids=current,
                               past_key_values=past, use_cache=True)
        past = output.past_key_values
        if step == 0:
            stats.prefill_time = time.perf_counter() - started

        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, token], dim=1)
        stats.generated_tokens += 1
        stats.ar_cycles += 1
        if token.item() in stops:
            break
        current = token

    stats.wall_time = time.perf_counter() - started
    return generated, stats


def _require_maskable_attention(model) -> None:
    """Refuse models whose attention would not apply the tree mask as written.

    Verification is equivalent to scoring each path separately only because the
    4-D mask hides everything but a node's ancestors, and two kinds of model
    break that. A kernel that drops custom masks (flash attention) lets the
    draft nodes see each other. And any per-layer restriction on what a token
    may attend to, such as a sliding window or an attention chunk, does not
    compose with our mask: transformers passes a 4-D mask through untouched, so ours
    replaces the restriction instead of narrowing it, and the tree pass attends
    further back than plain decoding does. Either way the accepted tokens
    quietly stop matching greedy decoding, so both are refused.
    """
    config = model.config
    implementation = getattr(config, "_attn_implementation", "eager")
    if implementation not in ("eager", "sdpa"):
        raise ValueError(
            f"tree verification needs an attention implementation that honours an "
            f"explicit mask; {implementation!r} does not. Load the model with "
            f"attn_implementation='sdpa'.")

    layer_types = getattr(config, "layer_types", None) or []
    window = getattr(config, "sliding_window", None)
    chunk = getattr(config, "attention_chunk_size", None)
    # When layer_types is present it is the authority; the scalar settings only
    # speak for models that do not list their layers.
    restricted = sorted(set(layer_types) - {"full_attention"}) or [
        name for name, value in (("sliding window", window and
                                  getattr(config, "use_sliding_window", True)),
                                 ("attention chunks", chunk))
        if value and not layer_types]
    if restricted:
        raise ValueError(
            f"this model restricts what each layer may attend to "
            f"({', '.join(restricted)}); the tree mask would replace that "
            f"restriction rather than narrow it, so verification would not "
            f"reproduce greedy decoding. Folding such a restriction into the "
            f"tree mask is not implemented.")


class _Session:
    """State of a single request: draft sources, KV cache and anchor.

    The anchor is the last token the model has committed but not yet attended
    to; it is the root of the next draft tree, and the cache holds everything
    before it.
    """

    def __init__(self, model, input_ids: torch.LongTensor, stops: frozenset,
                 config: GooseConfig, spine: Optional[SpineSource],
                 branches: Optional[BranchSource]) -> None:
        self.model = model
        self.config = config
        self.stops = stops
        self.device = input_ids.device

        self.tokens: List[int] = input_ids[0].tolist()
        self.prompt_length = len(self.tokens)
        self.stats = DecodeStats()

        self.spine = spine
        self.branches = branches

        self.past = None
        self.anchor = -1
        self.prev_anchor: Optional[int] = None
        self.spine_accept_ema = 0.5
        self.generated = 0
        self.finished = False

    def run(self, max_new_tokens: int,
            started: Optional[float] = None) -> Tuple[torch.LongTensor, DecodeStats]:
        started = time.perf_counter() if started is None else started
        self._prefill()
        while not self.finished and self.generated < max_new_tokens:
            self._cycle()
        self.stats.wall_time = time.perf_counter() - started

        # A cycle commits several tokens at once and can overshoot the limit.
        limit = self.prompt_length + max_new_tokens
        if len(self.tokens) > limit:
            self.stats.truncated_tokens = len(self.tokens) - limit
            self.stats.generated_tokens -= self.stats.truncated_tokens
            self.tokens = self.tokens[:limit]

        return torch.tensor([self.tokens], device=self.device, dtype=torch.long), self.stats

    def _cycle(self) -> None:
        """One cycle, routed by which sources have something to offer."""
        chain = self._context_chain()
        if self._has_branch_candidates():
            self._speculate(chain)
        elif chain:
            self._verify_chain(chain)
        else:
            self._step_autoregressive()

    def _context_chain(self) -> List[int]:
        if self.spine is None:
            return []
        return list(self.spine.propose(self.tokens))

    def _has_branch_candidates(self) -> bool:
        if self.branches is None:
            return False
        return self.branches.has_successors(
            self.anchor,
            prev_token=self.prev_anchor if self.config.use_bigram else None)

    def _speculate(self, chain: Sequence[int]) -> None:
        """Build a tree, verify it, commit the accepted path.

        With no context match the tree has no spine and this is a pure
        adjacency tree; with no adjacency data the cycle never gets here.
        """
        confidence = 0.0
        if chain and self.config.use_confidence:
            confidence = self.spine.confidence(self.tokens, chain)
            if confidence > CHAIN_EXTENSION_CONFIDENCE:
                chain = self._extend_chain(chain, confidence)

        if self.config.topology == "isotropic":
            tree = build_isotropic_tree(self.branches, self.anchor, self.prev_anchor,
                                        self.config, confidence)
        else:
            tree = build_spine_tree(self.branches, self.anchor, self.prev_anchor, chain,
                                    self.config, confidence=confidence,
                                    spine_accept_rate=self.spine_accept_ema)
        if tree.n_nodes < 2:
            # Every candidate fell below the score threshold.
            self._step_autoregressive()
            return

        result = verify_tree(self.model, tree, self.past, self.stats, self.device)
        self.past = result.past_key_values
        # Node 0's predecessor is prev_anchor, not None, but this route feeds
        # the bigram tier only with pairs it drafted itself; the anchor's own
        # bigram is left to the chain and autoregressive routes, as in the
        # research code.
        self._harvest(tree.tokens, result.logits,
                      [tree.tokens[p] if p >= 0 else None for p in tree.parents])
        self._account_for(tree, result.accepted_nodes)
        self._commit(result.accepted, result.bonus)

    def _verify_chain(self, chain: List[int]) -> None:
        """Verify a context match as a plain sequence.

        Reached when the adjacency table has nothing to say about the anchor,
        which is the state of the first cycles of a request. The tree route
        arrives at the same shape from the other side: a fully confident cycle
        spends its whole budget on the chain (Section 4.3).
        """
        drafted = [self.anchor] + chain
        output = timed_forward(
            self.model, self.stats,
            input_ids=torch.tensor([drafted], device=self.device, dtype=torch.long),
            past_key_values=self.past, use_cache=True)
        self.past = output.past_key_values

        predictions = output.logits[0].argmax(dim=-1).tolist()
        n_accepted = 0
        for i, token in enumerate(chain):
            if predictions[i] != token:
                break
            n_accepted = i + 1

        self._harvest(drafted, output.logits[0], [self.prev_anchor] + drafted[:-1])
        self.stats.chain_cycles += 1
        self.stats.draft_nodes += len(chain)
        self.stats.spine_nodes += len(chain)
        self.stats.spine_accepted += n_accepted

        self._commit(chain[:n_accepted], predictions[n_accepted])
        if not self.finished:
            # The cache still holds the rejected tail of the chain.
            self.past = crop_cache(self.past, len(self.tokens) - 1)

    def _step_autoregressive(self) -> None:
        """No source proposed anything: take one ordinary decoding step."""
        output = timed_forward(
            self.model, self.stats,
            input_ids=torch.tensor([[self.anchor]], device=self.device, dtype=torch.long),
            past_key_values=self.past, use_cache=True)
        self.past = output.past_key_values

        self._harvest([self.anchor], output.logits[0, -1:], [self.prev_anchor])
        self.stats.ar_cycles += 1
        self._commit([], int(output.logits[0, -1].argmax()), speculative=False)

    def _prefill(self) -> None:
        prefill_started = time.perf_counter()
        # Speculation needs a cache it can crop and reindex, so ask for one
        # explicitly rather than accept whatever default the model hands back.
        output = timed_forward(
            self.model, self.stats,
            input_ids=torch.tensor([self.tokens], device=self.device, dtype=torch.long),
            past_key_values=DynamicCache(), use_cache=True)
        self.past = output.past_key_values
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.stats.prefill_time = time.perf_counter() - prefill_started

        # The prefill scores every prompt position; all of it goes to the branch
        # source, which is what gives the first cycles anything to branch on.
        if self.config.use_prefill_harvest:
            processed = self.tokens[:output.logits.shape[1]]
            self._harvest(processed, output.logits[0], [None] + processed[:-1])
        if self.spine is not None:
            self.spine.on_prefill(self.tokens)

        anchor = int(output.logits[0, -1].argmax())
        self.tokens.append(anchor)
        self.generated += 1
        self.stats.generated_tokens += 1
        if self.spine is not None:
            self.spine.on_commit([anchor])
        self.prev_anchor = self.tokens[-2]
        self.anchor = anchor
        self.finished = anchor in self.stops

    def _extend_chain(self, chain: List[int], confidence: float) -> List[int]:
        """Keep proposing from the end of the draft while the evidence holds.

        A confident cycle is usually inside a passage the model is reproducing,
        where one proposal only reaches as far as the first n-gram it matched.
        The source is asked again from the end of the drafted text, and the
        extension stops when a proposal no longer earns full confidence.
        """
        limit = int(self.config.max_extended_chain * confidence)
        extended = list(chain)
        drafted = self.tokens + extended
        saturated = confidence >= 1.0

        while len(extended) < limit:
            following = self.spine.propose(drafted)
            if not following:
                break
            # The proposal is judged at the position it was made from, before
            # it is appended: a saturated cycle keeps growing only while each
            # further lookup itself earns full confidence.
            still_saturated = (not saturated
                               or self.spine.confidence(drafted, following) >= 1.0)
            extended.extend(following)
            drafted.extend(following)
            if not still_saturated:
                break
        return extended

    def _harvest(self, token_ids: Sequence[int], logits: torch.Tensor,
                 preceding: Sequence[Optional[int]]) -> None:
        if self.branches is None:
            return
        self.branches.observe(token_ids, logits,
                              preceding if self.config.use_bigram else None)

    def _account_for(self, tree: SpeculationTree, accepted_nodes: List[int]) -> None:
        spine_accepted = sum(1 for node in accepted_nodes
                             if tree.sources[node] == SPINE)
        branch_accepted = len(accepted_nodes) - spine_accepted

        if tree.n_spine_nodes > 0:
            self.stats.spine_tree_cycles += 1
            rate = spine_accepted / tree.n_spine_nodes
            alpha = self.config.ema_alpha
            self.spine_accept_ema = alpha * rate + (1.0 - alpha) * self.spine_accept_ema
        else:
            self.stats.branch_tree_cycles += 1

        if spine_accepted and branch_accepted:
            # The walk left the spine at the mismatch and kept going on a
            # branch: the synergy of Section 4.2.
            self.stats.spine_continuations += 1
            self.stats.spine_continuation_tokens += branch_accepted

        self.stats.draft_nodes += tree.n_nodes - 1   # the root is the anchor
        self.stats.spine_nodes += tree.n_spine_nodes
        self.stats.branch_nodes += tree.n_branch_nodes
        self.stats.spine_accepted += spine_accepted
        self.stats.branch_accepted += branch_accepted

    def _commit(self, accepted: List[int], bonus: int, speculative: bool = True) -> None:
        """Append the accepted tokens plus the bonus token, and move the anchor."""
        drafted = len(accepted)
        accepted, bonus, stop = _truncate_at_eos(accepted, bonus, self.stops)
        # Whatever followed a stop token in the same cycle was accepted by the
        # walk and then thrown away; the per-source counters above already
        # include it, so record the difference rather than lose it.
        self.stats.eos_dropped_tokens += drafted - len(accepted)
        self.tokens.extend(accepted)
        committed = len(accepted)

        if not (accepted and accepted[-1] in self.stops):
            self.tokens.append(bonus)
            committed += 1
            if speculative:
                self.stats.bonus_tokens += 1

        self.stats.generated_tokens += committed
        self.generated += committed
        if stop:
            self.finished = True
            return

        if self.spine is not None:
            self.spine.on_commit(accepted + [bonus])
        self.prev_anchor = accepted[-1] if accepted else self.anchor
        self.anchor = bonus


def _truncate_at_eos(accepted: List[int], bonus: int,
                     stops: frozenset) -> Tuple[List[int], int, bool]:
    """Cut the commit at the first end-of-sequence token, if any."""
    for i, token in enumerate(accepted):
        if token in stops:
            return accepted[:i + 1], token, True
    return accepted, bonus, bonus in stops
