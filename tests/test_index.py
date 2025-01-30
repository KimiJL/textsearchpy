import json
import pytest
from src.textsearchpy.index import Document, Index, IndexingError
from src.textsearchpy.query import (
    BooleanClause,
    BooleanQuery,
    PhraseQuery,
    TermQuery,
    WildcardQuery,
)
from src.textsearchpy.normalizers import StopwordsNormalizer
import os


def test_append_doc():
    index = Index()
    assert len(index.documents) == 0
    assert len(index.inverted_index) == 0
    assert index.total_tokens == 0

    doc1 = Document(
        text="repository contains all project files, including the revision history"
    )
    index.append([doc1])

    assert len(index.documents) == 1
    assert len(index.inverted_index) == 9
    assert index.total_tokens == 9

    doc2 = Document(text="repository repeats words words words")
    doc3 = Document(text="words and words and words")
    index.append([doc2, doc3])
    assert len(index.documents) == 3
    assert len(index.inverted_index) == 12
    assert index.total_tokens == 19


def test_append_doc_with_id():
    index = Index()
    doc1 = Document(text="i like cake", id="1")
    doc2 = Document(text="you like cookie", id="2")
    index.append([doc1, doc2])

    assert len(index) == 2
    assert index.documents["1"].text == "i like cake"
    assert index.documents["2"].text == "you like cookie"

    with pytest.raises(IndexingError):
        dupe_id_doc = Document(text="another doc", id="1")
        index.append([dupe_id_doc])


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

    # by default algo is not order sensitive
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
    doc5 = Document(text="cake like i")
    index.append([doc1, doc2, doc3, doc4, doc5])

    q = PhraseQuery(terms=["i", "like", "cake"], distance=0)
    docs = index.search(q)
    assert len(docs) == 2

    q = PhraseQuery(terms=["like", "i", "cake"], distance=0)
    docs = index.search(q)
    assert len(docs) == 2

    q = PhraseQuery(terms=["we", "like", "cake"], distance=0)
    docs = index.search(q)
    assert len(docs) == 1

    q = PhraseQuery(terms=["we", "like", "cake"], distance=1)
    docs = index.search(q)
    assert len(docs) == 1

    q = PhraseQuery(terms=["we", "like", "cake"], distance=2)
    docs = index.search(q)
    assert len(docs) == 2


def test_multi_term_phrase_ordered_query():
    index = Index()

    doc1 = Document(text="i like cake, but do we like this specific cake")
    doc2 = Document(text="you like cookie")
    doc3 = Document(text="we like cake")
    doc4 = Document(text="we should have a tea party")
    doc5 = Document(text="cake like i")
    index.append([doc1, doc2, doc3, doc4, doc5])

    q = PhraseQuery(terms=["like", "i", "cake"], distance=0, ordered=True)
    docs = index.search(q)
    assert len(docs) == 0

    q = PhraseQuery(terms=["i", "like", "cake"], distance=0, ordered=True)
    docs = index.search(q)
    assert len(docs) == 1

    q = PhraseQuery(terms=["we", "like", "cake"], distance=0, ordered=True)
    docs = index.search(q)
    assert len(docs) == 1

    q = PhraseQuery(terms=["we", "like", "cake"], distance=1, ordered=True)
    docs = index.search(q)
    assert len(docs) == 1

    q = PhraseQuery(terms=["we", "like", "cake"], distance=2, ordered=True)
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


def test_index_delete():
    index = Index()
    doc1 = Document(text="i like cake, but do we like this specific cake", id="1")
    doc2 = Document(text="you like cookie", id="2")
    doc3 = Document(text="we like cake", id="3")
    doc4 = Document(text="we should have a tea party", id="4")
    index.append([doc1, doc2, doc3, doc4])

    index.delete(ids=["1", "2", "3"])

    assert len(index) == 1
    assert index.inverted_index["we"] == ["4"]
    assert index.positional_index["we"] == {"4": [0]}
    assert index.total_tokens == 6

    assert len(index.search("cake")) == 0
    assert len(index.search("tea")) == 1


def test_query_with_filtered_tokens():
    index = Index(token_normalizers=[StopwordsNormalizer()])

    doc1 = Document(text="i like cake")
    index.append([doc1])

    # searching a term filtered by StopwordsNormalizer
    q = TermQuery(term="i")
    docs = index.search(q)
    assert len(docs) == 0

    # since like hits, assume the doc hits
    q = PhraseQuery(terms=["i", "like"], distance=0)
    docs = index.search(q)
    assert len(docs) == 1


def test_index_save_load(tmp_path):
    index = Index()
    doc1 = Document(text="you like cookie")
    doc2 = Document(text="we like cake")
    index.append([doc1, doc2])

    save_path = str(tmp_path / "test_save")
    index.save(path=save_path)

    index_file = os.path.join(save_path, "index.json")
    doc_file = os.path.join(save_path, "docs.jsonl")
    assert os.path.exists(index_file)
    assert os.path.exists(doc_file)

    with open(index_file, "r") as f:
        saved_index_file = json.load(f)

    assert saved_index_file["token_normalizers"] == ["LowerCaseNormalizer"]
    assert saved_index_file["tokenizer"] == "SimpleTokenizer"
    assert len(saved_index_file["inverted_index"]) == 5
    assert len(saved_index_file["positional_index"]) == 5

    saved_docs = []
    with open(doc_file, "r") as f:
        for line in f:
            saved_docs.append(json.loads(line))

    assert saved_docs[0]["text"] == "you like cookie"
    assert saved_docs[1]["text"] == "we like cake"

    new_index = Index()
    new_index.load_from_file(save_path)
    assert len(new_index.search("you")) == 1
    assert len(new_index.search("like")) == 2
    assert len(new_index.documents) == 2


def test_search_top_n():
    index = Index()
    doc1 = Document(text="i like cake, but do we like this specific cake")
    doc2 = Document(text="like like cake")
    doc3 = Document(text="cake cake cake")
    doc4 = Document(text="cake cake like")
    doc5 = Document(text="cake like i")
    index.append([doc1, doc2, doc3, doc4, doc5])

    docs = index.retrieve_top_n("cake")

    assert len(docs) == 5
    assert docs[0].text == doc3.text
    assert docs[1].text == doc4.text

    docs = index.retrieve_top_n("cake", n=2)

    assert len(docs) == 2
    assert docs[0].text == doc3.text
    assert docs[1].text == doc4.text

    docs = index.retrieve_top_n('"like cake"~1')
    assert len(docs) == 4

    docs = index.retrieve_top_n("i AND cake")
    assert len(docs) == 2


def test_wildcard_query():
    index = Index()

    doc1 = Document(text="i like cake, but do we like this specific cake")
    doc2 = Document(text="you like cookie")
    doc3 = Document(text="we like cake")
    doc4 = Document(text="we should have a tea party")
    doc5 = Document(text="we like cantaloupe")
    doc6 = Document(text="cape like i")
    index.append([doc1, doc2, doc3, doc4, doc5, doc6])

    query = WildcardQuery(term="ca?e")
    docs = index.search(query)
    assert len(docs) == 3

    query = WildcardQuery(term="ca*e")
    docs = index.search(query)
    assert len(docs) == 4

    # special case of no wildcard symbol should not break flow
    query = WildcardQuery(term="cake")
    docs = index.search(query)
    assert len(docs) == 2

    docs = index.search("c*e")
    assert len(docs) == 5
