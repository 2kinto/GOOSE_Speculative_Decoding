"""Verification must survive a model split across devices.

The paper's 33B cell runs on two GPUs with ``device_map="auto"``, which puts
different layers, and therefore different parts of the KV cache, on different
devices. The tree's attention mask, its position ids and the cache
surgery all have to reach the right one. A single-GPU machine can exercise the
same code path by sending half the stack to the CPU.

    GOOSE_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct python -m pytest tests/test_device_sharding.py
"""

import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from goose import GooseConfig, decode, decode_autoregressive

MODEL = os.environ.get("GOOSE_TEST_MODEL")
pytestmark = [
    pytest.mark.skipif(not MODEL, reason="set GOOSE_TEST_MODEL to run"),
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="needs a second device to split across"),
]

PROMPT = (
    "def add(self, data, flag=1):\n"
    "    result = self.calc.eval(data, flag)\n"
    "    return result\n\n"
    "def sub(self, data, flag=2):\n"
)

CONFIGS = {
    "goose": GooseConfig(),
    "isotropic": GooseConfig(topology="isotropic", branching_factor=3),
    "no bypass": GooseConfig(use_confidence=False),
}


@pytest.fixture(scope="module")
def split_model():
    """Half the decoder layers on the GPU, half on the CPU.

    Float32 throughout: half precision has no CPU kernel for these matmuls, and
    the point here is where tensors live, not what they weigh.
    """
    config = AutoConfig.from_pretrained(MODEL)
    n_layers = config.num_hidden_layers

    # Tied embeddings and head must share a device; everything else may move.
    device_map = {"model.embed_tokens": 0, "lm_head": 0, "model.rotary_emb": 0,
                  "model.norm": "cpu"}
    for i in range(n_layers):
        device_map[f"model.layers.{i}"] = 0 if i < n_layers // 2 else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map=device_map)

    placement = set(str(device) for device in model.hf_device_map.values())
    assert len(placement) > 1, f"model did not split: everything landed on {placement}"
    return model, tokenizer


@pytest.mark.parametrize("config", CONFIGS.values(), ids=list(CONFIGS))
def test_speculation_is_lossless_across_devices(split_model, config):
    model, tokenizer = split_model
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(0)

    reference, _ = decode_autoregressive(model, tokenizer, input_ids, max_new_tokens=32)
    expected = reference[0, input_ids.shape[1]:].tolist()

    output, stats = decode(model, tokenizer, input_ids, max_new_tokens=32, config=config)

    assert output[0, input_ids.shape[1]:].tolist() == expected
    assert stats.forward_calls < stats.generated_tokens
