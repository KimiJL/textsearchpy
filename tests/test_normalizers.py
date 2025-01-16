from src.textsearchpy.normalizers import (
    LowerCaseNormalizer,
    StopwordsNormalizer,
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
