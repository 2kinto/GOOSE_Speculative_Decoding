from goose.context_match import match_confidence
from goose.tree import BRANCH, ROOT, SPINE, SpeculationTree
from goose.verify import greedy_walk


def tree_from(edges, sources):
    """Build a tree from ``[(token, parent), ...]`` plus a source per node."""
    tokens, parents, depths, children = [], [], [], []
    for node, (token, parent) in enumerate(edges):
        tokens.append(token)
        parents.append(parent)
        depths.append(0 if parent < 0 else depths[parent] + 1)
        children.append([])
        if parent >= 0:
            children[parent].append(node)
    spine_end = 1 + sum(1 for source in sources if source == SPINE)
    return SpeculationTree(tokens, parents, depths, list(sources), children, spine_end)


def spine_tree():
    """Anchor 0, spine 11 -> 12, and a branch 21 -> 22 forking at the anchor."""
    return tree_from(
        [(10, -1), (11, 0), (12, 1), (21, 0), (22, 3)],
        [ROOT, SPINE, SPINE, BRANCH, BRANCH],
    )


def test_walk_follows_the_spine_while_the_model_agrees():
    tree = spine_tree()
    predictions = [11, 12, 99, 0, 0]
    path, last = greedy_walk(tree, predictions)
    assert path == [1, 2]
    assert last == 2


def test_walk_stops_at_the_first_mismatch():
    tree = spine_tree()
    predictions = [99, 0, 0, 0, 0]
    assert greedy_walk(tree, predictions) == ([], 0)


def test_walk_takes_a_branch_when_the_spine_is_wrong():
    tree = spine_tree()
    predictions = [21, 0, 0, 22, 0]
    path, last = greedy_walk(tree, predictions)
    assert path == [3, 4]
    assert last == 4


def test_walk_continues_on_a_branch_after_the_spine_breaks():
    # Spine 11 accepted, then the model wants 31, which only a branch of the
    # spine node offers: the spine continuation of Section 4.2.
    tree = tree_from(
        [(10, -1), (11, 0), (12, 1), (31, 1), (32, 3)],
        [ROOT, SPINE, SPINE, BRANCH, BRANCH],
    )
    predictions = [11, 31, 0, 32, 0]
    path, last = greedy_walk(tree, predictions)
    assert [tree.tokens[node] for node in path] == [11, 31, 32]
    assert last == 4


def test_spine_wins_when_both_sources_offer_the_same_token():
    # The branch child is listed first, so a walk that simply took the first
    # match would pick it; the spine has to win on source, not on order.
    tree = tree_from(
        [(10, -1), (11, 0), (11, 0)],
        [ROOT, BRANCH, SPINE],
    )
    path, _ = greedy_walk(tree, [11, 0, 0])
    assert tree.sources[path[0]] == SPINE
    assert path[0] == 2


def test_confidence_saturates_on_consensus_and_grows_with_length():
    assert match_confidence(0, consensus=False) == 0.0
    assert match_confidence(3, consensus=True) == 1.0
    assert match_confidence(4, consensus=False) < match_confidence(6, consensus=False)
    assert match_confidence(8, consensus=False) >= 0.7
    assert match_confidence(50, consensus=False) <= 0.9
