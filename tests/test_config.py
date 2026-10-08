import pytest

from goose import GooseConfig


@pytest.mark.parametrize("fields", [
    {"node_budget": 1},                       # no room for the root and a draft
    {"node_budget": 0},
    {"max_branch_depth": 0},
    {"adjacency_top_k": 0},
    {"branching_factor": 0},
    {"min_score": 1.0},                       # would prune every candidate
    {"min_score": -0.1},
    {"spine_branch_ratio": 1.5},
    {"ema_alpha": -0.1},
    {"ngram_lengths": ()},
    {"ngram_lengths": (0, 1)},
    {"max_chain": 0},
    {"max_extended_chain": 10, "max_chain": 20},
    {"long_match": 0},
    {"spine_ratio_tiers": (0.5, 0.5)},
    {"topology": "balanced"},
    {"use_context_match": False, "use_adjacency": False},
])
def test_impossible_configurations_are_refused(fields):
    with pytest.raises(ValueError):
        GooseConfig(**fields)


def test_the_paper_configuration_is_accepted():
    config = GooseConfig()
    assert config.node_budget == 60
    assert config.budget_extension == 40
    assert config.spine_ratio(0.9) == 0.50
    assert config.spine_ratio(0.3) == 0.30
    assert config.spine_ratio(0.1) == 0.15
