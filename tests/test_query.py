from src.textsearchpy.query import BooleanQuery, PhraseQuery, TermQuery, parse_query


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
