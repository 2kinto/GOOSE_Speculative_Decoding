"""The decoder refuses models whose attention would not apply the tree mask."""

from types import SimpleNamespace

import pytest

from goose.decoding import _require_maskable_attention


def model_with(**fields):
    fields.setdefault("_attn_implementation", "sdpa")
    return SimpleNamespace(config=SimpleNamespace(**fields))


def test_sliding_window_layers_are_refused():
    # Gemma-2/3 style: the layer list names the restricted layers.
    with pytest.raises(ValueError, match="restricts what each layer"):
        _require_maskable_attention(model_with(
            layer_types=["full_attention", "sliding_attention"], sliding_window=4096))


def test_a_window_without_a_layer_list_is_refused():
    # Mistral style: a scalar window and no per-layer types.
    with pytest.raises(ValueError, match="sliding window"):
        _require_maskable_attention(model_with(sliding_window=4096))


def test_chunked_attention_is_refused():
    # Llama-4 style: chunks rather than a window, and the same failure mode.
    # transformers hands a 4-D mask through untouched, so ours would replace
    # the chunk restriction instead of narrowing it.
    with pytest.raises(ValueError, match="restricts what each layer"):
        _require_maskable_attention(model_with(
            layer_types=["chunked_attention", "full_attention"],
            attention_chunk_size=8192))


def test_a_disabled_window_is_accepted():
    # Qwen2.5/Qwen3 carry sliding-window fields but switch them off; the paper's
    # Qwen3-8B is this case and must keep working.
    _require_maskable_attention(model_with(sliding_window=32768,
                                           use_sliding_window=False))
    _require_maskable_attention(model_with(sliding_window=None))
    _require_maskable_attention(model_with(layer_types=["full_attention"] * 4,
                                           sliding_window=4096))


def test_flash_attention_is_refused():
    with pytest.raises(ValueError, match="honours an explicit mask"):
        _require_maskable_attention(model_with(_attn_implementation="flash_attention_2",
                                               sliding_window=None))
