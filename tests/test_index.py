from src.index import tokenize, Document, Index


def test_tokenize():
    text = "Iteratively yield tokens as unicode strings, removing accent marks and optionally lowercasing the unidoce string by assigning True to one of the parameters, lowercase, to_lower, or lower."
    tokens = tokenize(text)
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
    tokens = tokenize(text)
    assert tokens == ["Version", "was", "released", "on", "October"]


def test_append_doc():
    index = Index()
    assert len(index.documents) == 0
    assert len(index.inverted_index) == 0

    doc1 = Document(
        text="repository contains all project files, including the revision history"
    )
    index.append([doc1])

    assert len(index.documents) == 1
    assert len(index.inverted_index) == 9

    doc2 = Document(text="repository repeats words words words")
    doc3 = Document(text="words and words and words")
    index.append([doc2, doc3])
    assert len(index.documents) == 3
    assert len(index.inverted_index) == 12


def test_search():
    index = Index()

    doc1 = Document(text="i like cake")
    doc2 = Document(text="you like cookie")
    doc3 = Document(text="we like cake")

    index.append([doc1, doc2, doc3])

    docs = index.search("like")
    assert len(docs) == 3

    docs = index.search("you")
    assert len(docs) == 1
    assert docs[0].text == "you like cookie"

    docs = index.search("cake")
    assert len(docs) == 2
