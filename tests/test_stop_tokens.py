from types import SimpleNamespace

from goose import stop_tokens


def model_with(eos):
    return SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=eos))


def tokenizer_with(eos):
    return SimpleNamespace(eos_token_id=eos)


def test_every_declared_stop_token_is_honoured():
    # Llama-3-8B-Instruct declares <|end_of_text|> and <|eot_id|>; the tokenizer
    # names only the second, and a run that honours just that one keeps going
    # after the model has finished.
    stops = stop_tokens(model_with([128001, 128009]), tokenizer_with(128009))
    assert stops == {128001, 128009}


def test_a_single_declared_stop_token_still_works():
    assert stop_tokens(model_with(2), tokenizer_with(2)) == {2}


def test_the_tokenizer_is_the_fallback():
    assert stop_tokens(model_with(None), tokenizer_with(7)) == {7}


def test_the_caller_can_override():
    stops = stop_tokens(model_with([1, 2]), tokenizer_with(1), eos_token_id=99)
    assert stops == {99}


def test_an_override_may_be_a_list():
    assert stop_tokens(model_with(None), tokenizer_with(1), eos_token_id=[4, 5]) == {4, 5}
