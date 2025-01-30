from src.textsearchpy.query import (
    BooleanClause,
    BooleanQuery,
    PhraseQuery,
    TermQuery,
    WildcardQuery,
    parse_query,
)


def test_parse_term_query():
    query = "word"
    q = parse_query(query)
    assert isinstance(q, TermQuery)
    assert q.term == "word"


def test_basic_boolean_query():
    query = "word search"
    q = parse_query(query)
    assert isinstance(q, BooleanQuery)
    assert len(q.clauses) == 2
    assert isinstance(q.clauses[0].query, TermQuery)
    assert q.clauses[0].query.term == "word"
    assert q.clauses[0].clause == "SHOULD"
    assert isinstance(q.clauses[1].query, TermQuery)
    assert q.clauses[1].query.term == "search"
    assert q.clauses[1].clause == "SHOULD"

    query = "word AND search"
    q = parse_query(query)
    assert isinstance(q, BooleanQuery)
    assert len(q.clauses) == 2
    assert isinstance(q.clauses[0].query, TermQuery)
    assert q.clauses[0].query.term == "word"
    assert q.clauses[0].clause == "MUST"
    assert isinstance(q.clauses[1].query, TermQuery)
    assert q.clauses[1].query.term == "search"
    assert q.clauses[1].clause == "MUST"

    query = "word NOT search"
    q = parse_query(query)
    assert isinstance(q, BooleanQuery)
    assert len(q.clauses) == 2
    assert isinstance(q.clauses[0].query, TermQuery)
    assert q.clauses[0].query.term == "word"
    assert q.clauses[0].clause == "SHOULD"
    assert isinstance(q.clauses[1].query, TermQuery)
    assert q.clauses[1].query.term == "search"
    assert q.clauses[1].clause == "MUST_NOT"


def test_compound_boolean_query():
    query = "word AND search NOT found"
    q = parse_query(query)
    assert isinstance(q, BooleanQuery)
    assert len(q.clauses) == 3
    assert isinstance(q.clauses[0].query, TermQuery)
    assert q.clauses[0].query.term == "word"
    assert q.clauses[0].clause == "MUST"
    assert isinstance(q.clauses[1].query, TermQuery)
    assert q.clauses[1].query.term == "search"
    assert q.clauses[1].clause == "MUST"
    assert isinstance(q.clauses[2].query, TermQuery)
    assert q.clauses[2].query.term == "found"
    assert q.clauses[2].clause == "MUST_NOT"

    query = "word OR search NOT found"
    q = parse_query(query)
    assert isinstance(q, BooleanQuery)
    assert len(q.clauses) == 3
    assert isinstance(q.clauses[0].query, TermQuery)
    assert q.clauses[0].query.term == "word"
    assert q.clauses[0].clause == "SHOULD"
    assert isinstance(q.clauses[1].query, TermQuery)
    assert q.clauses[1].query.term == "search"
    assert q.clauses[1].clause == "SHOULD"
    assert isinstance(q.clauses[2].query, TermQuery)
    assert q.clauses[2].query.term == "found"
    assert q.clauses[2].clause == "MUST_NOT"


def test_basic_phrase_query():
    query = '"word search"'
    q = parse_query(query)
    assert isinstance(q, PhraseQuery)
    assert q.terms == ["word", "search"]
    assert q.distance == 0

    query = '"word search"~5'
    q = parse_query(query)
    assert isinstance(q, PhraseQuery)
    assert q.terms == ["word", "search"]
    assert q.distance == 5


def test_multi_term_phrase_query():
    query = '"multi word search"~3'
    q = parse_query(query)
    assert isinstance(q, PhraseQuery)
    assert q.terms == ["multi", "word", "search"]
    assert q.distance == 3


def test_basic_group_query():
    query = "(group word) AND search"
    q = parse_query(query)
    assert isinstance(q, BooleanQuery)
    assert isinstance(q.clauses[0].query, BooleanQuery)
    assert q.clauses[0].clause == "MUST"
    assert isinstance(q.clauses[1].query, TermQuery)
    assert q.clauses[1].clause == "MUST"

    sub_q = q.clauses[0].query
    assert isinstance(sub_q.clauses[1].query, TermQuery)
    assert sub_q.clauses[0].clause == "SHOULD"
    assert isinstance(sub_q.clauses[1].query, TermQuery)
    assert sub_q.clauses[1].clause == "SHOULD"


def test_wildcard_query():
    query = "sea?ch"
    q = parse_query(query)
    assert isinstance(q, WildcardQuery)
    assert q.term == query

    query = "sea*ch"
    q = parse_query(query)
    assert isinstance(q, WildcardQuery)
    assert q.term == query


def test_to_query_string():
    query = "(group word) AND search"
    q = parse_query(query)

    assert q.to_query_string() == "(group OR word) AND search"

    query = '("this is"~10 OR word NOT engine) AND search NOT ("complex at all"~3)'
    q = parse_query(query)
    assert (
        q.to_query_string()
        == '("this is"~10 OR word NOT engine) AND search NOT ("complex at all"~3)'
    )

    term_1 = TermQuery(term="fox")
    term_2 = TermQuery(term="dog")
    query_left = BooleanQuery(
        clauses=[
            BooleanClause(query=term_1, clause="SHOULD"),
            BooleanClause(query=term_2, clause="SHOULD"),
        ]
    )

    term_3 = TermQuery(term="quick")
    term_4 = TermQuery(term="lazy")
    query_right = BooleanQuery(
        clauses=[
            BooleanClause(query=term_3, clause="SHOULD"),
            BooleanClause(query=term_4, clause="MUST_NOT"),
        ]
    )

    query = BooleanQuery(
        clauses=[
            BooleanClause(query=query_left, clause="MUST"),
            BooleanClause(query=query_right, clause="MUST"),
        ]
    )
    assert query.to_query_string() == "(fox OR dog) AND (quick NOT lazy)"
