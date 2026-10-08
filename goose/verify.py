"""Verification of a draft tree in a single forward pass (Section 4.2).

Scoring a whole tree in one pass, with a mask that shows each node only its own
ancestors, is tree-structured verification (Miao et al., 2024; Cai et al., 2024).
What is verified here is the anisotropic tree of Section 4.1, and the walk that
reads the result is Algorithm 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from .runtime import cache_length, select_cache, timed_forward
from .stats import DecodeStats
from .tree import SPINE, SpeculationTree


@dataclass
class Verification:
    """Outcome of one verification cycle."""

    accepted: List[int]        # accepted tokens, in order
    accepted_nodes: List[int]  # their node indices in the tree
    bonus: int                 # argmax at the last accepted position, free with this pass
    past_key_values: object
    logits: torch.Tensor       # [n_nodes, vocab_size], for the adjacency harvest


def greedy_walk(tree: SpeculationTree, predictions: List[int]) -> Tuple[List[int], int]:
    """Longest path the target model agrees with (Algorithm 3).

    From the root, a child is accepted when its token is what the model
    predicts at the parent. Spine children are checked before branch children,
    so a cycle that could continue on either source stays on the spine; because
    a token is drafted at most once per parent, this priority only ever settles
    ties and cannot change which tokens are accepted.

    Returns the accepted path and the node the walk stopped at, whose
    prediction becomes the bonus token.
    """
    node = 0
    path: List[int] = []
    while True:
        prediction = predictions[node]
        matched = None
        for child in tree.children[node]:
            if tree.tokens[child] != prediction:
                continue
            if tree.sources[child] == SPINE:
                matched = child      # the spine wins outright
                break
            if matched is None:
                matched = child      # otherwise the first branch that matches
        if matched is None:
            return path, node
        path.append(matched)
        node = matched


def verify_tree(model, tree: SpeculationTree, past_key_values, stats: DecodeStats,
                device: torch.device) -> Verification:
    """Score every node of ``tree`` at once and walk the accepted path.

    The tree is fed as a flat sequence. Two things make that equivalent to
    scoring each root-to-leaf path on its own: the attention mask, which hides
    everything but a node's ancestors, and the position ids, which are tree
    depths rather than sequence offsets.
    """
    n_nodes = tree.n_nodes
    prefix_length = cache_length(past_key_values)
    dtype = next(model.parameters()).dtype

    mask = torch.full((1, 1, n_nodes, prefix_length + n_nodes),
                      torch.finfo(dtype).min, device=device, dtype=dtype)
    mask[:, :, :, :prefix_length] = 0.0
    rows, cols = tree.ancestor_pairs()
    mask[0, 0,
         torch.tensor(rows, device=device),
         torch.tensor(cols, device=device) + prefix_length] = 0.0

    depths = torch.tensor(tree.depths, device=device, dtype=torch.long)
    output = timed_forward(
        model, stats,
        input_ids=torch.tensor([tree.tokens], device=device, dtype=torch.long),
        position_ids=(depths + prefix_length).unsqueeze(0),
        attention_mask=mask,
        past_key_values=past_key_values,
        use_cache=True,
    )

    predictions = output.logits[0].argmax(dim=-1).tolist()
    path, last_node = greedy_walk(tree, predictions)

    keep = torch.empty(prefix_length + 1 + len(path), dtype=torch.long, device=device)
    keep[:prefix_length] = torch.arange(prefix_length, device=device)
    keep[prefix_length] = prefix_length          # the anchor, i.e. the tree root
    if path:
        keep[prefix_length + 1:] = torch.tensor(
            [prefix_length + node for node in path], device=device)

    return Verification(
        accepted=[tree.tokens[node] for node in path],
        accepted_nodes=path,
        bonus=predictions[last_node],
        past_key_values=select_cache(output.past_key_values, keep),
        logits=output.logits[0],
    )
