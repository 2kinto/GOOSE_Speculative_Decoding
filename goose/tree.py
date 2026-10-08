"""Draft-tree construction (Section 4.1, Algorithm 1).

The spine tree is anisotropic: context-matched tokens, which the target model
accepts several times more often, are laid out as one deep chain, and adjacency
successors, cheap but far less reliable, fork off it as branches whose width
decays with depth. The two sources are the ones prior training-free work uses
separately: n-gram matches against the context (Saxena, 2023) and the adjacency
table of Token Recycling (Luo et al., 2025).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import GooseConfig
from .sources import BranchSource

ROOT = "root"
SPINE = "spine"
BRANCH = "branch"


@dataclass
class SpeculationTree:
    """A draft tree flattened into one sequence; node 0 is the anchor.

    ``sources[i]`` records which draft source produced node ``i``. The label is
    used at construction time and to break ties during the greedy walk; it
    never enters the acceptance test itself, which is token identity against
    the target model's argmax.

    Spine nodes occupy indices ``1 .. spine_end - 1``.
    """

    tokens: List[int]
    parents: List[int]
    depths: List[int]
    sources: List[str]
    children: List[List[int]]
    spine_end: int = 1

    @property
    def n_nodes(self) -> int:
        return len(self.tokens)

    @property
    def n_spine_nodes(self) -> int:
        return self.spine_end - 1

    @property
    def n_branch_nodes(self) -> int:
        return len(self.tokens) - self.spine_end

    def ancestor_pairs(self) -> Tuple[List[int], List[int]]:
        """(row, column) index pairs of the tree attention mask.

        Node ``i`` attends to itself and to its ancestors, and to nothing else,
        so every root-to-node path is scored as if it were the only draft.
        """
        rows: List[int] = []
        cols: List[int] = []
        for node in range(len(self.tokens)):
            rows.append(node)
            cols.append(node)
            ancestor = self.parents[node]
            while ancestor >= 0:
                rows.append(node)
                cols.append(ancestor)
                ancestor = self.parents[ancestor]
        return rows, cols


class _TreeBuilder:
    """Accumulates nodes; keeps the parent/child/depth bookkeeping in one place."""

    def __init__(self, anchor: int) -> None:
        self.tokens = [anchor]
        self.parents = [-1]
        self.depths = [0]
        self.sources = [ROOT]
        self.children: List[List[int]] = [[]]

    def add(self, token: int, parent: int, source: str) -> int:
        node = len(self.tokens)
        self.tokens.append(token)
        self.parents.append(parent)
        self.depths.append(self.depths[parent] + 1)
        self.sources.append(source)
        self.children.append([])
        self.children[parent].append(node)
        return node

    @property
    def n_nodes(self) -> int:
        return len(self.tokens)

    def finish(self, spine_end: int) -> SpeculationTree:
        return SpeculationTree(
            tokens=self.tokens,
            parents=self.parents,
            depths=self.depths,
            sources=self.sources,
            children=self.children,
            spine_end=spine_end,
        )


def build_spine_tree(
    adjacency: BranchSource,
    anchor: int,
    prev_token: Optional[int],
    chain: Optional[Sequence[int]],
    config: GooseConfig,
    confidence: float = 0.0,
    spine_accept_rate: float = 0.5,
) -> SpeculationTree:
    """Build the anisotropic spine tree of Algorithm 1.

    ``chain`` is the context match; an empty chain leaves a pure adjacency tree,
    which is the shape a cycle with no context match falls back to.
    ``confidence`` (Section 4.3) both enlarges the budget and shifts it towards
    the spine, so a fully confident cycle spends everything on one long chain.
    """
    budget = config.node_budget + int(confidence * config.budget_extension)
    base_ratio = config.spine_ratio(spine_accept_rate)
    ratio = base_ratio + confidence * (1.0 - base_ratio)

    chain = list(chain or [])
    spine_length = min(len(chain), int(budget * ratio)) if chain else 0

    builder = _TreeBuilder(anchor)

    # Step 1: the spine, one context-matched token per node.
    parent = 0
    for token in chain[:spine_length]:
        parent = builder.add(token, parent, SPINE)
    spine_end = builder.n_nodes
    spine_nodes = list(range(1, spine_end))

    branch_budget = budget - 1 - spine_length
    if spine_length > 0 and branch_budget > 0:
        spine_share = max(1, int(branch_budget * config.spine_branch_ratio))
        root_share = branch_budget - spine_share
    else:
        spine_share, root_share = 0, branch_budget

    # Nodes whose subtree may still grow, with the score they were drafted at.
    # A node joins only while its depth is below the limit, so the frontier
    # itself carries the depth bound.
    frontier: deque = deque()

    # Step 2: alternatives to the first spine token, attached at the root.
    first_spine_token = chain[0] if chain else None
    attached = 0
    for token, score in _successors(adjacency, anchor, prev_token, config):
        if attached >= root_share or score < config.min_score:
            break
        if token == first_spine_token:
            continue
        node = builder.add(token, 0, BRANCH)
        attached += 1
        if builder.depths[node] < config.max_branch_depth:
            frontier.append((node, score))

    # Step 3: alternatives at each spine node, wide near the root and narrow
    # deeper down. Proposition 2 asks for a linear decay; the 1/i rule keeps
    # that monotone shape without having to estimate acceptance rates online.
    if spine_share > 0 and spine_nodes:
        for position, allowance in enumerate(_harmonic_split(spine_share, len(spine_nodes))):
            if allowance <= 0:
                continue
            node = spine_nodes[position]
            successor_of = builder.tokens[node]
            preceding = builder.tokens[builder.parents[node]]
            next_spine_token = (builder.tokens[spine_nodes[position + 1]]
                                if position + 1 < len(spine_nodes) else None)
            attached = 0
            for token, score in _successors(adjacency, successor_of, preceding, config):
                if attached >= allowance:
                    break
                if token == next_spine_token:
                    continue          # already on the spine; drafting it twice wastes a node
                if score < config.min_score:
                    break
                child = builder.add(token, node, BRANCH)
                attached += 1
                if builder.depths[child] < config.max_branch_depth:
                    frontier.append((child, score))

    # Step 4: extend branches breadth-first through the adjacency table
    # (Algorithm 1, Step 4). Width follows the score a branch was
    # drafted at, so confident branches grow into chains and doubtful ones stop
    # at one node.
    remaining = budget - builder.n_nodes
    while frontier and remaining > 0:
        node, score = frontier.popleft()
        depth = builder.depths[node]

        grandparent = builder.parents[node]
        preceding = builder.tokens[grandparent] if grandparent >= 0 else None
        successors = _successors(adjacency, builder.tokens[node], preceding, config)
        width = max(1, min(len(successors), math.ceil(remaining * score)))
        width = min(width, max(1, config.adjacency_top_k // depth))

        for token, child_score in successors[:width]:
            if remaining <= 0 or child_score < config.min_score:
                break
            child = builder.add(token, node, BRANCH)
            remaining -= 1
            if depth + 1 < config.max_branch_depth:
                frontier.append((child, child_score))

    return builder.finish(spine_end)


def build_isotropic_tree(
    adjacency: BranchSource,
    anchor: int,
    prev_token: Optional[int],
    config: GooseConfig,
    confidence: float = 0.0,
) -> SpeculationTree:
    """Balanced k-ary tree over the same draft source and the same budget.

    This is the topology control of Section 5.3: every node gets the same
    number of children, so the tree carries no assumption about which candidate
    is more likely to survive.
    """
    budget = config.node_budget + int(confidence * config.budget_extension)
    builder = _TreeBuilder(anchor)
    remaining = budget - 1
    frontier: deque = deque([0])

    while frontier and remaining > 0:
        node = frontier.popleft()
        depth = builder.depths[node]

        if node == 0:
            preceding = prev_token
        else:
            grandparent = builder.parents[node]
            preceding = builder.tokens[grandparent] if grandparent >= 0 else None

        attached = 0
        for token, score in _successors(adjacency, builder.tokens[node], preceding, config):
            if attached >= config.branching_factor or remaining <= 0:
                break
            if score < config.min_score:
                break
            child = builder.add(token, node, BRANCH)
            attached += 1
            remaining -= 1
            if depth + 1 < config.max_branch_depth:
                frontier.append(child)

    return builder.finish(spine_end=1)


def _harmonic_split(budget: int, n_positions: int) -> List[int]:
    """Split ``budget`` children over ``n_positions`` spine nodes with weights 1/i."""
    weights = [1.0 / (i + 1) for i in range(n_positions)]
    total = sum(weights)

    allocation: List[int] = []
    remaining = budget
    for weight in weights:
        if remaining <= 0:
            allocation.append(0)
            continue
        share = min(max(1, int(budget * weight / total)), remaining)
        allocation.append(share)
        remaining -= share
    return allocation


def _successors(adjacency: BranchSource, token: int, prev_token: Optional[int],
                config: GooseConfig) -> List[Tuple[int, float]]:
    return adjacency.successors(
        token,
        prev_token=prev_token if config.use_bigram else None,
        limit=config.adjacency_top_k,
    )
