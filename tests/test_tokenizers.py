from src.textsearchpy.tokenizers import SimpleTokenizer, NGramTokenizer


def test_simple_tokenizer():
    text = "Iteratively yield tokens as unicode strings, removing accent marks and optionally lowercasing the unidoce string by assigning True to one of the parameters, lowercase, to_lower, or lower."
    tokenizer = SimpleTokenizer()
    tokens = tokenizer.tokenize(text)
    assert tokens == [
        "Iteratively",
        "yield",
        "tokens",
        "as",
        "unicode",
        "strings",
        "removing",
        "accent",
        "marks",
        "and",
        "optionally",
        "lowercasing",
        "the",
        "unidoce",
        "string",
        "by",
        "assigning",
        "True",
        "to",
        "one",
        "of",
        "the",
        "parameters",
        "lowercase",
        "to_lower",
        "or",
        "lower",
    ]

    text = "Version 4.0 was released on October 12, 2012."
    tokens = tokenizer.tokenize(text)
    assert tokens == ["Version", "was", "released", "on", "October"]


def test_ngram_tokenizer():
    tokenizer = NGramTokenizer(min_gram=5, max_gram=5)
    text = "Quick Fox"
    tokens = tokenizer.tokenize(text)

    assert tokens == ["Quick", "uick ", "ick F", "ck Fo", "k Fox"]

    tokenizer = NGramTokenizer(min_gram=5, max_gram=6)
    tokens = tokenizer.tokenize(text)
    assert tokens == [
        "Quick",
        "Quick ",
        "uick ",
        "uick F",
        "ick F",
        "ick Fo",
        "ck Fo",
        "ck Fox",
        "k Fox",
    ]
