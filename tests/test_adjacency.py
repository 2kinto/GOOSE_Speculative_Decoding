import torch

from goose.adjacency import AdjacencyTable

VOCAB = 32


def logits_favouring(*token_ids):
    """Logits whose ranking is exactly the given tokens, best first."""
    row = torch.full((VOCAB,), -20.0)
    for rank, token in enumerate(token_ids):
        row[token] = 10.0 - rank
    return row


def test_successors_come_back_ranked():
    table = AdjacencyTable(VOCAB, top_k=4, device="cpu")
    table.harvest([5], logits_favouring(7, 8, 9, 10).unsqueeze(0))

    successors = table.successors(5)
    assert [token for token, _ in successors] == [7, 8, 9, 10]
    scores = [score for _, score in successors]
    assert scores == sorted(scores, reverse=True)


def test_unseen_token_has_no_successors():
    table = AdjacencyTable(VOCAB, top_k=4, device="cpu")
    table.harvest([5], logits_favouring(7, 8).unsqueeze(0))
    assert table.successors(6) == []


def test_bigram_entry_wins_over_unigram():
    # Two harvests of the same token with different predecessors and different
    # successors: the bigram entry has to be the one that answers, so the two
    # tiers must be able to disagree.
    table = AdjacencyTable(VOCAB, top_k=2, device="cpu")
    table.harvest([5], logits_favouring(7, 8).unsqueeze(0), prev_tokens=[3])
    table.harvest([5], logits_favouring(20, 21).unsqueeze(0), prev_tokens=[9])

    assert [token for token, _ in table.successors(5, prev_token=3)] == [7, 8]
    assert [token for token, _ in table.successors(5, prev_token=9)] == [20, 21]
    # An unseen predecessor falls back to the unigram tier, which the last
    # harvest left behind.
    assert [token for token, _ in table.successors(5, prev_token=4)] == [20, 21]
    assert [token for token, _ in table.successors(5)] == [20, 21]


def test_limit_truncates_the_result():
    table = AdjacencyTable(VOCAB, top_k=8, device="cpu")
    table.harvest([5], logits_favouring(1, 2, 3, 4, 5, 6, 7, 8).unsqueeze(0))
    assert len(table.successors(5, limit=3)) == 3


def test_a_token_repeated_in_one_batch_keeps_its_last_statistics():
    # A draft tree holds the same token at several nodes and a prompt repeats
    # tokens constantly, so one harvest scores a token more than once. Writing
    # those rows without de-duplicating resolves the collision arbitrarily, and
    # separately for the ids and for the scores, which can leave an entry
    # holding one position's successors next to another position's scores. The
    # batch here is large enough to exercise the batched write path.
    generator = torch.Generator().manual_seed(0)
    vocab, top_k, batch = 512, 10, 400
    token_ids = torch.randint(0, vocab, (batch,), generator=generator).tolist()
    logits = torch.randn(batch, vocab, generator=generator) * 4.0

    table = AdjacencyTable(vocab, top_k, device="cpu")
    table.harvest(token_ids, logits)

    for token in set(token_ids):
        position = max(i for i, t in enumerate(token_ids) if t == token)
        values, ids = torch.topk(logits[position], top_k)
        scores = torch.softmax(values, dim=-1).to(torch.float16)

        stored = table.successors(token)
        assert [i for i, _ in stored] == ids.tolist(), f"token {token}: wrong successors"
        assert [round(s, 4) for _, s in stored] == [round(float(v), 4) for v in scores], (
            f"token {token}: scores do not belong to the successors stored with them")


def test_harvest_counts_only_the_rows_it_could_use():
    table = AdjacencyTable(VOCAB, top_k=2, device="cpu")
    assert table.harvest([], torch.empty(0, VOCAB)) == 0
    assert table.harvest([1, 2], logits_favouring(3, 4).unsqueeze(0)) == 1
