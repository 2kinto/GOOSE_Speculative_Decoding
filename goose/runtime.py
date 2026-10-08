"""Forward passes and KV-cache surgery.

Speculation reuses one cache across cycles: the tree is appended to it, and
after verification everything but the accepted path is dropped again.
"""

from __future__ import annotations

import time

import torch

from .stats import DecodeStats


def timed_forward(model, stats: DecodeStats, **kwargs):
    """Run the target model and charge the call to ``stats``.

    The synchronisation makes ``model_time`` a real GPU measurement rather than
    the time it takes to queue the kernels; every method in the benchmark pays
    for it identically.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stats.model_time += time.perf_counter() - started
    stats.forward_calls += 1
    return output


def cache_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    return past_key_values.get_seq_length()


def crop_cache(past_key_values, length: int):
    """Keep the first ``length`` positions."""
    if past_key_values is not None:
        past_key_values.crop(length)
    return past_key_values


def select_cache(past_key_values, positions: torch.Tensor):
    """Keep the given positions, in the given order."""
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        for layer in layers:
            index = positions.to(layer.keys.device)
            layer.keys = layer.keys.index_select(2, index)
            layer.values = layer.values.index_select(2, index)
        return past_key_values

    # transformers < 4.54 exposes the cache as two parallel lists.
    keys, values = past_key_values.key_cache, past_key_values.value_cache
    for i in range(len(keys)):
        index = positions.to(keys[i].device)
        keys[i] = keys[i].index_select(2, index)
        values[i] = values[i].index_select(2, index)
    return past_key_values
