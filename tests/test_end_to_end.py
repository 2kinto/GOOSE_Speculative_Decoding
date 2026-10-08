"""Speculation must reproduce greedy decoding exactly.

Needs a checkpoint, so it only runs when one is named:

    GOOSE_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct python -m pytest tests/test_end_to_end.py

A small model on the CPU is enough; the property under test does not depend on
the model, and a mismatch here means a bug in the tree, the mask or the walk.
"""

import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from goose import GooseConfig, decode, decode_autoregressive

MODEL = os.environ.get("GOOSE_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set GOOSE_TEST_MODEL to run")

# Repetitive on purpose: it exercises the spine, the branches and the
# continuation from one to the other within a few dozen tokens.
PROMPT = (
    "def add(self, data, flag=1):\n"
    "    result = self.calc.eval(data, flag)\n"
    "    return result\n\n"
    "def sub(self, data, flag=2):\n"
)

CONFIGS = {
    "goose": GooseConfig(),
    "chain only": GooseConfig(use_adjacency=False),
    "no spine": GooseConfig(use_context_match=False),
    "no spine branches": GooseConfig(spine_branch_ratio=0.0),
    "no bigram": GooseConfig(use_bigram=False),
    "no bypass": GooseConfig(use_confidence=False),
    "no prefill harvest": GooseConfig(use_prefill_harvest=False),
    "isotropic": GooseConfig(topology="isotropic", branching_factor=3),
}


@pytest.fixture(scope="module")
def target():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def greedy_output(target):
    model, tokenizer = target
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)
    output, _ = decode_autoregressive(model, tokenizer, input_ids, max_new_tokens=48)
    return output[0, input_ids.shape[1]:].tolist()


@pytest.mark.parametrize("config", CONFIGS.values(), ids=list(CONFIGS))
def test_output_matches_greedy_decoding(target, greedy_output, config):
    model, tokenizer = target
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)

    output, stats = decode(model, tokenizer, input_ids, max_new_tokens=48, config=config)

    assert output[0, input_ids.shape[1]:].tolist() == greedy_output
    assert stats.forward_calls < stats.generated_tokens


def test_a_budget_below_one_token_is_refused(target):
    model, tokenizer = target
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)

    # A caller computing a budget arithmetically can land on zero; silently
    # cutting the prompt back and reporting a negative token count would be
    # worse than saying no.
    for budget in (0, -3):
        with pytest.raises(ValueError):
            decode(model, tokenizer, input_ids, max_new_tokens=budget)
        with pytest.raises(ValueError):
            decode_autoregressive(model, tokenizer, input_ids, max_new_tokens=budget)


def test_counters_account_for_every_committed_token_when_it_stops_early(target):
    """The accounting identity has to survive a stop inside an accepted path.

    The plain test below never reaches an end-of-sequence token, so it cannot
    see the tokens a walk accepted and a stop then discarded. Here the run is
    repeated with a stop token taken from several points of its own output.
    """
    model, tokenizer = target
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)

    output, _ = decode(model, tokenizer, input_ids, max_new_tokens=48,
                       config=CONFIGS["goose"])
    generated = output[0, input_ids.shape[1]:].tolist()

    dropped_somewhere = False
    for position in range(1, len(generated)):
        _, stats = decode(model, tokenizer, input_ids, max_new_tokens=48,
                          eos_token_id=generated[position], config=CONFIGS["goose"])
        committed = (1
                     + stats.spine_accepted + stats.branch_accepted
                     + stats.bonus_tokens + stats.ar_cycles)
        assert committed == stats.committed_tokens, (
            f"stopping on the token at position {position} breaks the accounting")
        dropped_somewhere |= stats.eos_dropped_tokens > 0

    assert dropped_somewhere, "no stop point cut an accepted path; the test proves nothing"


def test_counters_account_for_every_committed_token(target, config=CONFIGS["goose"]):
    """The per-source counters must reconcile with the tokens actually emitted.

    One prefill token, then each cycle commits its accepted drafts plus one
    token the model produced itself: the bonus on a speculative cycle, the single
    step on an autoregressive one. A cycle can overshoot the limit, and
    what the limit cut is `truncated_tokens`.
    """
    model, tokenizer = target
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)

    _, stats = decode(model, tokenizer, input_ids, max_new_tokens=32, config=config)

    committed = (1
                 + stats.spine_accepted + stats.branch_accepted
                 + stats.bonus_tokens + stats.ar_cycles)
    assert committed == stats.committed_tokens
    assert stats.draft_nodes == stats.spine_nodes + stats.branch_nodes


def test_the_confidence_switch_really_gates_the_confidence_path(target, monkeypatch):
    """With use_confidence=False the confidence machinery must never run.

    The repetitive prompt makes the default configuration consult it on almost
    every cycle, so tripwiring the function proves the switch both ways.
    """
    import goose.context_match as context_match

    model, tokenizer = target
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)

    def tripwire(*args, **kwargs):
        raise AssertionError("match_confidence was consulted")

    monkeypatch.setattr(context_match, "match_confidence", tripwire)

    # Off: the tripwire must never fire.
    decode(model, tokenizer, input_ids, max_new_tokens=24,
           config=GooseConfig(use_confidence=False))

    # On: this prompt must reach it.
    with pytest.raises(AssertionError, match="match_confidence was consulted"):
        decode(model, tokenizer, input_ids, max_new_tokens=24, config=GooseConfig())
