from src.textsearchpy.normalizers import (
    LowerCaseNormalizer,
    StopwordsNormalizer,
    NGramNormalizer,
)


def test_lowercase_normalizer():
    normalizer = LowerCaseNormalizer()

    tokens = ["Hi", "JOHN", "what"]
    tokens = normalizer.normalize(tokens)

    assert tokens == ["hi", "john", "what"]


def test_stopwords_normalizer():
    normalizer = StopwordsNormalizer()

    tokens = ["what", "is", "your", "name"]
    t_tokens = normalizer.normalize(tokens)
    assert t_tokens == ["name"]

    tokens = ["rare", "search"]
    t_tokens = normalizer.normalize(tokens)
    assert tokens == t_tokens


def test_stopwrods_normalizer_custom_words():
    stopwords = ["mock"]
    normalizer = StopwordsNormalizer(stopwords=stopwords)

    tokens = ["this", "is", "mock", "testing"]
    t_tokens = normalizer.normalize(tokens)
    assert t_tokens == ["this", "is", "testing"]


def test_ngram_normalizer():
    normalizer = NGramNormalizer(min_gram=3, max_gram=5, preserve_original=False)

    tokens = ["this", "is", "testing"]
    t_tokens = normalizer.normalize(tokens)

    assert t_tokens == [
        "thi",
        "this",
        "his",
        "is",
        "tes",
        "test",
        "testi",
        "est",
        "esti",
        "estin",
        "sti",
        "stin",
        "sting",
        "tin",
        "ting",
        "ing",
    ]


def test_ngram_normalizer_preserve_original():
    normalizer = NGramNormalizer(min_gram=2, max_gram=3, preserve_original=True)
    tokens = ["this", "works"]
    t_tokens = normalizer.normalize(tokens)

    assert t_tokens == [
        "this",
        "th",
        "thi",
        "hi",
        "his",
        "is",
        "works",
        "wo",
        "wor",
        "or",
        "ork",
        "rk",
        "rks",
        "ks",
    ]


def test_ngram_normalizer_same_min_max():
    normalizer = NGramNormalizer(min_gram=4, max_gram=4, preserve_original=False)

    tokens = ["this", "is", "works"]
    t_tokens = normalizer.normalize(tokens)

    assert t_tokens == ["this", "is", "work", "orks"]
