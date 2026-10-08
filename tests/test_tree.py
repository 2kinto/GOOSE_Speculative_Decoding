import torch

from goose.adjacency import AdjacencyTable
from goose.config import GooseConfig
from goose.tree import BRANCH, SPINE, build_isotropic_tree, build_spine_tree

VOCAB = 256


def adjacency_with(successors, top_k=4):
    """Table whose top-K successors of each key are exactly the listed tokens."""
    table = AdjacencyTable(VOCAB, top_k, device="cpu")
    tokens, rows = [], []
    for token, ids in successors.items():
        logits = torch.full((VOCAB,), -20.0)
        for rank, successor in enumerate(ids):
            logits[successor] = 5.0 - rank
        tokens.append(token)
        rows.append(logits)
    table.harvest(tokens, torch.stack(rows))
    return table


def uniform_adjacency(top_k=4):
    """Every token has the same four successors, so branches can always grow."""
    return adjacency_with({token: [200, 201, 202, 203] for token in range(VOCAB)}, top_k)


def children_of(tree, node, source=None):
    return [c for c in tree.children[node]
            if source is None or tree.sources[c] == source]


def test_spine_is_a_chain_of_the_context_match():
    config = GooseConfig()
    chain = list(range(1, 11))
    tree = build_spine_tree(uniform_adjacency(), anchor=0, prev_token=None,
                            chain=chain, config=config)

    assert tree.n_spine_nodes == len(chain)
    spine = [node for node in range(tree.n_nodes) if tree.sources[node] == SPINE]
    assert [tree.tokens[node] for node in spine] == chain
    assert [tree.parents[node] for node in spine] == [0] + spine[:-1]


def test_budget_is_respected():
    config = GooseConfig(node_budget=24)
    tree = build_spine_tree(uniform_adjacency(), anchor=0, prev_token=None,
                            chain=list(range(1, 30)), config=config)
    assert tree.n_nodes <= config.node_budget


def test_spine_is_capped_by_its_share_of_the_budget():
    config = GooseConfig(node_budget=20)
    tree = build_spine_tree(uniform_adjacency(), anchor=0, prev_token=None,
                            chain=list(range(1, 30)), config=config)
    # Default tier at the neutral acceptance estimate: half the budget.
    assert tree.n_spine_nodes == 10


def test_branch_width_decays_along_the_spine():
    config = GooseConfig()
    tree = build_spine_tree(uniform_adjacency(), anchor=0, prev_token=None,
                            chain=list(range(1, 11)), config=config)

    spine = [node for node in range(tree.n_nodes) if tree.sources[node] == SPINE]
    widths = [len(children_of(tree, node, BRANCH)) for node in spine]
    assert widths == sorted(widths, reverse=True)
    assert widths[0] > widths[-1]


def test_spine_successor_is_not_drafted_twice():
    # The adjacency proposes exactly the token the spine already carries.
    table = adjacency_with({token: [token + 1, 250, 251] for token in range(VOCAB - 1)})
    tree = build_spine_tree(table, anchor=0, prev_token=None,
                            chain=[1, 2, 3, 4], config=GooseConfig())

    for node in range(tree.n_nodes):
        drafted = [tree.tokens[c] for c in tree.children[node]]
        assert len(drafted) == len(set(drafted))


def test_confidence_gives_the_whole_budget_to_the_spine():
    config = GooseConfig()
    budget = config.node_budget + config.budget_extension
    tree = build_spine_tree(uniform_adjacency(), anchor=0, prev_token=None,
                            chain=list(range(1, budget + 20)), config=config,
                            confidence=1.0)

    # A fully confident cycle is the linear bypass: one long chain, no branches.
    assert tree.n_spine_nodes == budget
    assert tree.n_branch_nodes == 0
    # The root is not charged against a spine that took the whole budget, so
    # such a cycle carries one node more than the budget names. Pinned here
    # because it is the research code's arithmetic.
    assert tree.n_nodes == budget + 1


def test_low_acceptance_shifts_the_budget_to_the_branches():
    config = GooseConfig()
    chain = list(range(1, 31))
    confident = build_spine_tree(uniform_adjacency(), anchor=0, prev_token=None,
                                 chain=chain, config=config, spine_accept_rate=0.9)
    doubtful = build_spine_tree(uniform_adjacency(), anchor=0, prev_token=None,
                                chain=chain, config=config, spine_accept_rate=0.1)
    assert doubtful.n_spine_nodes < confident.n_spine_nodes


def test_branch_extension_stops_at_the_depth_limit():
    # The limit bounds how far a branch is grown, not where one may be
    # attached: a spine longer than the limit carries branches below it, and
    # those are simply never extended.
    config = GooseConfig(max_branch_depth=3)
    tree = build_spine_tree(uniform_adjacency(), anchor=0, prev_token=None,
                            chain=list(range(1, 11)), config=config)

    for node in range(1, tree.n_nodes):
        parent = tree.parents[node]
        if tree.sources[node] == BRANCH and tree.sources[parent] == BRANCH:
            assert tree.depths[parent] < config.max_branch_depth, (
                f"node {node} was grown from a branch already at depth "
                f"{tree.depths[parent]}")

    below_limit = [node for node in range(tree.n_nodes)
                   if tree.sources[node] == BRANCH
                   and tree.depths[node] > config.max_branch_depth]
    assert below_limit, "the fixture should place branches past the limit"
    assert all(not tree.children[node] for node in below_limit)


def test_low_scoring_successors_are_pruned():
    # Second successor carries almost no probability mass.
    logits = torch.full((VOCAB,), -20.0)
    logits[40] = 10.0
    logits[41] = 0.0
    table = AdjacencyTable(VOCAB, 2, device="cpu")
    table.harvest([0], logits.unsqueeze(0))

    tree = build_spine_tree(table, anchor=0, prev_token=None, chain=None,
                            config=GooseConfig())
    assert [tree.tokens[c] for c in tree.children[0]] == [40]


def test_empty_chain_leaves_a_pure_branch_tree():
    tree = build_spine_tree(uniform_adjacency(), anchor=0, prev_token=None,
                            chain=None, config=GooseConfig())
    assert tree.n_spine_nodes == 0
    assert tree.n_branch_nodes == tree.n_nodes - 1


def test_isotropic_tree_has_uniform_width():
    config = GooseConfig(topology="isotropic", branching_factor=3, node_budget=40)
    tree = build_isotropic_tree(uniform_adjacency(), anchor=0, prev_token=None,
                                config=config)

    assert tree.n_nodes <= config.node_budget
    internal = [node for node in range(tree.n_nodes) if tree.children[node]]
    assert {len(tree.children[node]) for node in internal} == {3}


def test_mask_lets_every_node_see_exactly_its_ancestors():
    tree = build_spine_tree(uniform_adjacency(), anchor=0, prev_token=None,
                            chain=[1, 2, 3], config=GooseConfig())
    rows, cols = tree.ancestor_pairs()

    visible = {node: set() for node in range(tree.n_nodes)}
    for row, col in zip(rows, cols):
        visible[row].add(col)

    for node in range(tree.n_nodes):
        expected = {node}
        ancestor = tree.parents[node]
        while ancestor >= 0:
            expected.add(ancestor)
            ancestor = tree.parents[ancestor]
        assert visible[node] == expected
