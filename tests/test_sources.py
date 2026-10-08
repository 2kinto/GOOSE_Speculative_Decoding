"""The draft sources are parameters: anything speaking the two protocols works."""

from types import SimpleNamespace

import torch

from goose import AdjacencyTable, BranchSource, ContextMatcher, SpineSource
from goose.config import GooseConfig
from goose.tree import build_spine_tree


def test_the_default_sources_satisfy_the_protocols():
    assert isinstance(ContextMatcher(), SpineSource)
    assert isinstance(AdjacencyTable(64, 4, device="cpu"), BranchSource)


class ScriptedSpine:
    """A spine source that proposes whatever it is told to."""

    def __init__(self, chain, confidence=0.0):
        self.chain, self._confidence = list(chain), confidence
        self.seen = []

    def on_prefill(self, tokens):
        self.seen.append(("prefill", list(tokens)))

    def on_commit(self, tokens):
        self.seen.append(("commit", list(tokens)))

    def propose(self, tokens):
        return list(self.chain)

    def confidence(self, tokens, chain):
        return self._confidence


class ScriptedBranches:
    """A branch source with a fixed successor list for every token."""

    def __init__(self, successors):
        self.successors_of = successors
        self.observed = 0

    def observe(self, token_ids, logits, prev_tokens=None):
        self.observed += len(token_ids)
        return len(token_ids)

    def successors(self, token_id, prev_token=None, limit=None):
        return list(self.successors_of)[:limit]

    def has_successors(self, token_id, prev_token=None):
        return bool(self.successors_of)


def test_a_custom_branch_source_feeds_the_tree_builder():
    branches = ScriptedBranches([(200, 0.6), (201, 0.3), (202, 0.1)])
    tree = build_spine_tree(branches, anchor=0, prev_token=None, chain=[1, 2, 3],
                            config=GooseConfig(node_budget=20))
    assert tree.n_spine_nodes == 3
    assert tree.n_branch_nodes > 0
    assert set(tree.tokens[tree.spine_end:]) <= {200, 201, 202}


def test_the_decoder_accepts_swapped_sources(monkeypatch):
    """The loop must drive whatever sources it is given, and only those."""
    import goose.decoding as decoding

    # A stub model whose forward returns logits favouring one fixed token, so
    # the run is deterministic and needs no checkpoint.
    vocab = 32
    stop = 31

    class Output:
        def __init__(self, n):
            self.logits = torch.full((1, n, vocab), -10.0)
            self.logits[0, :, 7] = 10.0
            self.past_key_values = None

    class Model:
        config = SimpleNamespace(vocab_size=vocab, _attn_implementation="eager",
                                 layer_types=None, sliding_window=None)
        generation_config = SimpleNamespace(eos_token_id=stop)

        def __call__(self, input_ids=None, **kwargs):
            return Output(input_ids.shape[1])

        def parameters(self):
            yield torch.zeros(1)

    monkeypatch.setattr(decoding, "verify_tree", None)   # must not be reached below
    monkeypatch.setattr(decoding, "cache_length", lambda cache: 0, raising=False)

    spine = ScriptedSpine(chain=[7, 7, 7], confidence=0.0)
    branches = ScriptedBranches([])       # nothing: forces the chain route
    tokenizer = SimpleNamespace(eos_token_id=stop)

    # crop_cache is called on the chain route; the stub cache is None.
    monkeypatch.setattr(decoding, "crop_cache", lambda cache, n: cache)

    output, stats = decoding.decode(Model(), tokenizer, torch.tensor([[1, 2, 3]]),
                                    max_new_tokens=8, spine_source=spine,
                                    branch_source=branches)
    assert stats.chain_cycles >= 1
    assert spine.seen[0][0] == "prefill"
    assert any(kind == "commit" for kind, _ in spine.seen)
    assert branches.observed > 0        # every forward was reported to the branch source


def test_a_source_that_ignores_the_protocol_is_rejected_by_the_type_check():
    assert not isinstance(object(), SpineSource)
    assert not isinstance(object(), BranchSource)


def test_a_saturated_extension_judges_each_proposal_where_it_was_made():
    """The stop rule of the confident-chain extension (Section 4.3).

    A saturated cycle keeps growing only while each further proposal itself
    earns full confidence, and that confidence is the source's verdict at the
    position the proposal was made from, not at the position after it has been
    appended. Judged after appending, the loop would stop one round too
    early on this stream.
    """
    import goose.decoding as decoding
    from goose.config import GooseConfig
    from goose.context_match import ContextMatcher

    A, B, C, x, y, z = 1, 2, 3, 4, 5, 6
    matcher = ContextMatcher((3,), max_continuation=2)
    # "A B C" always continues with x (consensus), while "C x y" continues
    # with A twice and z once (no consensus).
    matcher.extend([A, B, C, x, y, A, B, C, x, y, A, B, C, x, y, z])

    class Session:
        spine = matcher
        tokens = [A, B, C, x, y, A, B, C, x, y, A, B, C, x, y, z, A, B, C]
        config = GooseConfig(max_extended_chain=100, max_chain=2)

    first = matcher.propose(Session.tokens)
    assert first == [x, y]
    assert matcher.confidence(Session.tokens, first) == 1.0

    extended = decoding._Session._extend_chain(Session(), first, 1.0)
    # Round 1: from "A B C" (consensus) -> append x y. Round 2: from "C x y"
    # (no consensus) -> its proposal is still appended, then the loop stops.
    assert extended == [x, y, A, B]


def test_the_prefill_harvest_switch_gates_the_prompt_seeding():
    """use_prefill_harvest=False must leave the table empty until decoding starts.

    The prompt is what gives the first cycles anything to branch on, so the
    switch is checked by counting what reaches the branch source before the
    first commit, in both positions.
    """
    import torch
    from types import SimpleNamespace

    import goose.decoding as decoding
    from goose.config import GooseConfig

    vocab, stop = 32, 31

    class Output:
        def __init__(self, n):
            self.logits = torch.full((1, n, vocab), -10.0)
            self.logits[0, :, 7] = 10.0
            self.past_key_values = None

    class Model:
        config = SimpleNamespace(vocab_size=vocab, _attn_implementation="eager",
                                 layer_types=None, sliding_window=None)
        generation_config = SimpleNamespace(eos_token_id=stop)

        def __call__(self, input_ids=None, **kwargs):
            return Output(input_ids.shape[1])

        def parameters(self):
            yield torch.zeros(1)

    tokenizer = SimpleNamespace(eos_token_id=stop)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])

    seen = {}
    for label, harvest in (("on", True), ("off", False)):
        branches = ScriptedBranches([])
        decoding.decode(Model(), tokenizer, prompt, max_new_tokens=1,
                        spine_source=ScriptedSpine(chain=[]),
                        branch_source=branches,
                        config=GooseConfig(use_prefill_harvest=harvest))
        seen[label] = branches.observed

    assert seen["on"] >= prompt.shape[1], "the prompt's own positions should be harvested"
    assert seen["off"] == 0, "nothing should reach the branch source before decoding"
