from goose.context_match import ContextMatcher


def matcher(tokens, ngram_lengths=(3,), max_continuation=20):
    engine = ContextMatcher(ngram_lengths, max_continuation)
    engine.extend(tokens)
    return engine


def test_match_copies_the_earlier_continuation():
    engine = matcher([1, 2, 3, 7, 8, 9, 4, 5, 1, 2, 3])
    assert engine.match([1, 2, 3]) == [7, 8, 9, 4, 5, 1, 2, 3]


def test_match_is_capped_at_max_continuation():
    engine = matcher(list(range(100)) + [0, 1, 2], max_continuation=5)
    assert engine.match([0, 1, 2]) == [3, 4, 5, 6, 7]


def test_no_match_returns_empty():
    assert matcher([1, 2, 3]).match([4, 5, 6]) == []


def test_longer_ngrams_are_preferred():
    # The 3-gram (1,2,3) first occurs at index 1 and continues with 50, while
    # the 4-gram (9,1,2,3) first occurs at index 5 and continues with 100. The
    # two therefore disagree, and the longer context has to decide.
    engine = matcher([0, 1, 2, 3, 50, 9, 1, 2, 3, 100, 8, 1, 2, 3],
                     ngram_lengths=(4, 3))
    assert engine.match([1, 2, 3])[0] == 50
    assert engine.match([9, 1, 2, 3])[0] == 100


def test_consensus_needs_two_agreeing_witnesses():
    single = matcher([1, 2, 3, 42, 5, 1, 2, 3])
    assert single.consensus([1, 2, 3]) is False

    agreeing = matcher([1, 2, 3, 42, 5, 1, 2, 3, 42, 6, 1, 2, 3])
    assert agreeing.consensus([1, 2, 3]) is True

    disagreeing = matcher([1, 2, 3, 42, 5, 1, 2, 3, 43, 6, 1, 2, 3])
    assert disagreeing.consensus([1, 2, 3]) is False


def test_consensus_falls_back_to_shorter_ngrams():
    # The 5-gram of the query occurs only where the query itself sits, which is
    # no evidence at all; the 3-gram inside it has two earlier occurrences and
    # both continue with the same token.
    engine = matcher([4, 5, 6, 7, 9, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6],
                     ngram_lengths=(5, 4, 3))
    assert engine.match([2, 3, 4, 5, 6]) == [7, 9, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6]
    assert engine.consensus([2, 3, 4, 5, 6]) is True


def test_the_query_position_is_not_its_own_witness():
    # One earlier occurrence plus the query itself is a single witness.
    engine = matcher([1, 2, 3, 8, 9, 1, 2, 3], ngram_lengths=(3,))
    assert engine.consensus([1, 2, 3]) is False


def test_generated_tokens_join_the_pool():
    engine = matcher([1, 2, 3, 7, 8])
    assert engine.match([7, 8, 9]) == []
    engine.extend([9, 4, 5, 6])
    assert engine.match([7, 8, 9]) == [4, 5, 6]
