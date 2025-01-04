from src.textsearchpy.index import SimpleTokenizer, Document, Index
from src.textsearchpy.query import BooleanClause, BooleanQuery, PhraseQuery, TermQuery


def test_tokenize():
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


def test_append_doc_mixed_type():
    index = Index()

    doc1 = Document(text="abcd")
    doc2 = "qwer"

    index.append([doc1, doc2])
    assert len(index.documents) == 2


def test_positional_index_append():
    index = Index()

    doc1 = Document(text="this book has a lot of words for a book")
    doc2 = Document(text="can you give this book away for me")

    index.append([doc1, doc2])

    assert len(index.positional_index.keys()) == 13
    assert len(index.positional_index["book"].keys()) == 2
    assert len(index.positional_index["a"].keys()) == 1
    assert len(index.positional_index["away"].keys()) == 1

    # get the doc_id
    doc1_id = list(index.positional_index["a"].keys())[0]
    doc2_id = list(index.positional_index["away"].keys())[0]

    assert index.documents[doc1_id].text == doc1.text
    assert index.documents[doc2_id].text == doc2.text

    assert index.positional_index["book"][doc1_id] == [1, 9]
    assert index.positional_index["book"][doc2_id] == [4]


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

    docs = index.search("what")
    assert len(docs) == 0


def test_term_search():
    index = Index()

    doc1 = Document(text="i like cake")
    doc2 = Document(text="you like cookie")
    doc3 = Document(text="we like cake")

    index.append([doc1, doc2, doc3])

    q = TermQuery(term="cookie")
    docs = index.search(q)
    assert len(docs) == 1

    q = TermQuery(term="cake")
    docs = index.search(q)
    assert len(docs) == 2


def test_boolean_search():
    index = Index()

    doc1 = Document(text="i like cake")
    doc2 = Document(text="you like cookie")
    doc3 = Document(text="we like cake")
    doc4 = Document(text="we should have a tea party")

    index.append([doc1, doc2, doc3, doc4])

    q1 = TermQuery(term="cookie")
    q2 = TermQuery(term="cake")
    q = BooleanQuery(
        clauses=[
            BooleanClause(query=q1, clause="SHOULD"),
            BooleanClause(query=q2, clause="SHOULD"),
        ]
    )
    docs = index.search(q)
    assert len(docs) == 3

    q1 = TermQuery(term="like")
    q2 = TermQuery(term="we")
    q = BooleanQuery(
        clauses=[
            BooleanClause(query=q1, clause="MUST"),
            BooleanClause(query=q2, clause="MUST"),
        ]
    )
    docs = index.search(q)
    assert len(docs) == 1

    q1 = TermQuery(term="cake")
    q2 = TermQuery(term="like")
    q = BooleanQuery(
        clauses=[
            BooleanClause(query=q1, clause="MUST_NOT"),
            BooleanClause(query=q2, clause="SHOULD"),
        ]
    )
    docs = index.search(q)
    assert len(docs) == 1

    q1 = TermQuery(term="cake")
    q2 = TermQuery(term="cookie")
    q = BooleanQuery(
        clauses=[
            BooleanClause(query=q1, clause="MUST"),
            BooleanClause(query=q2, clause="SHOULD"),
        ]
    )
    docs = index.search(q)
    assert len(docs) == 2


def test_phrase_query():
    index = Index()

    doc1 = Document(text="i like cake, but do we like this specific cake")
    doc2 = Document(text="you like cookie")
    doc3 = Document(text="we like cake")
    doc4 = Document(text="we should have a tea party")
    index.append([doc1, doc2, doc3, doc4])

    q = PhraseQuery(terms=["like", "cake"], distance=0)
    docs = index.search(q)
    assert len(docs) == 2

    q = PhraseQuery(terms=["we", "cake"], distance=1)
    docs = index.search(q)
    assert len(docs) == 1

    # currently algo is not order sensitive
    q = PhraseQuery(terms=["cake", "like"], distance=0)
    docs = index.search(q)
    assert len(docs) == 2

    q = PhraseQuery(terms=["we", "cake"], distance=2)
    docs = index.search(q)
    assert len(docs) == 2

    q = PhraseQuery(terms=["we", "cake"], distance=0)
    docs = index.search(q)
    assert len(docs) == 0

    q = PhraseQuery(terms=["we", "cookie"], distance=0)
    docs = index.search(q)
    assert len(docs) == 0


def test_phrase_query_with_same_word():
    index = Index()
    doc = Document(text="you like cookie")
    index.append([doc])
    # should not match because there is not two like tokens
    q = PhraseQuery(terms=["like", "like"], distance=0)
    docs = index.search(q)
    assert len(docs) == 0

    doc2 = Document(text="you like like cookie")
    index.append([doc2])
    docs = index.search(q)
    assert len(docs) == 1


def test_phrase_query_ordered():
    index = Index()
    doc = Document(text="you like cookie")
    index.append([doc])

    q = PhraseQuery(terms=["cookie", "you"], distance=1, ordered=False)
    docs = index.search(q)
    assert len(docs) == 1

    q = PhraseQuery(terms=["cookie", "you"], distance=1, ordered=True)
    docs = index.search(q)
    assert len(docs) == 0

    q = PhraseQuery(terms=["you", "cookie"], distance=1, ordered=True)
    docs = index.search(q)
    assert len(docs) == 1


def test_multi_term_phrase_query():
    index = Index()

    doc1 = Document(text="i like cake, but do we like this specific cake")
    doc2 = Document(text="you like cookie")
    doc3 = Document(text="we like cake")
    doc4 = Document(text="we should have a tea party")
    index.append([doc1, doc2, doc3, doc4])

    q = PhraseQuery(terms=["i", "like", "cake"], distance=0)
    docs = index.search(q)
    assert len(docs) == 1

    q = PhraseQuery(terms=["we", "like", "cake"], distance=0)
    docs = index.search(q)
    assert len(docs) == 1

    q = PhraseQuery(terms=["we", "like", "cake"], distance=1)
    docs = index.search(q)
    assert len(docs) == 1

    q = PhraseQuery(terms=["we", "like", "cake"], distance=2)
    docs = index.search(q)
    assert len(docs) == 2


def test_string_query():
    index = Index()
    doc1 = Document(text="i like cake, but do we like this specific cake")
    doc2 = Document(text="you like cookie")
    doc3 = Document(text="we like cake")
    doc4 = Document(text="we should have a tea party")
    index.append([doc1, doc2, doc3, doc4])

    q = "tea"
    docs = index.search(q)
    assert len(docs) == 1

    q = "cake cookie"
    docs = index.search(q)
    assert len(docs) == 3

    q = "cake AND specific"
    docs = index.search(q)
    assert len(docs) == 1

    q = "cake NOT like"
    docs = index.search(q)
    assert len(docs) == 0


def test_index_length():
    index = Index()
    assert len(index) == 0

    doc1 = Document(text="you like cookie")
    doc2 = Document(text="we like cake")
    index.append([doc1, doc2])

    assert len(index) == 2
