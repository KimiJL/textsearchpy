from src.query import BooleanQuery, PhraseQuery, TermQuery, parse_query


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
    assert isinstance(q.clauses[0][0], TermQuery)
    assert q.clauses[0][0].term == "word"
    assert q.clauses[0][1] == "SHOULD"
    assert isinstance(q.clauses[1][0], TermQuery)
    assert q.clauses[1][0].term == "search"
    assert q.clauses[1][1] == "SHOULD"

    query = "word AND search"
    q = parse_query(query)
    assert isinstance(q, BooleanQuery)
    assert len(q.clauses) == 2
    assert isinstance(q.clauses[0][0], TermQuery)
    assert q.clauses[0][0].term == "word"
    assert q.clauses[0][1] == "MUST"
    assert isinstance(q.clauses[1][0], TermQuery)
    assert q.clauses[1][0].term == "search"
    assert q.clauses[1][1] == "MUST"

    query = "word NOT search"
    q = parse_query(query)
    assert isinstance(q, BooleanQuery)
    assert len(q.clauses) == 2
    assert isinstance(q.clauses[0][0], TermQuery)
    assert q.clauses[0][0].term == "word"
    assert q.clauses[0][1] == "SHOULD"
    assert isinstance(q.clauses[1][0], TermQuery)
    assert q.clauses[1][0].term == "search"
    assert q.clauses[1][1] == "MUST_NOT"


def test_compound_boolean_query():
    query = "word AND search NOT found"
    q = parse_query(query)
    assert isinstance(q, BooleanQuery)
    assert len(q.clauses) == 3
    assert isinstance(q.clauses[0][0], TermQuery)
    assert q.clauses[0][0].term == "word"
    assert q.clauses[0][1] == "MUST"
    assert isinstance(q.clauses[1][0], TermQuery)
    assert q.clauses[1][0].term == "search"
    assert q.clauses[1][1] == "MUST"
    assert isinstance(q.clauses[2][0], TermQuery)
    assert q.clauses[2][0].term == "found"
    assert q.clauses[2][1] == "MUST_NOT"

    query = "word OR search NOT found"
    q = parse_query(query)
    assert isinstance(q, BooleanQuery)
    assert len(q.clauses) == 3
    assert isinstance(q.clauses[0][0], TermQuery)
    assert q.clauses[0][0].term == "word"
    assert q.clauses[0][1] == "SHOULD"
    assert isinstance(q.clauses[1][0], TermQuery)
    assert q.clauses[1][0].term == "search"
    assert q.clauses[1][1] == "SHOULD"
    assert isinstance(q.clauses[2][0], TermQuery)
    assert q.clauses[2][0].term == "found"
    assert q.clauses[2][1] == "MUST_NOT"


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


def test_basic_group_query():
    query = "(group word) AND search"
    q = parse_query(query)
    assert isinstance(q, BooleanQuery)
    assert isinstance(q.clauses[0][0], BooleanQuery)
    assert q.clauses[0][1] == "MUST"
    assert isinstance(q.clauses[1][0], TermQuery)
    assert q.clauses[1][1] == "MUST"

    sub_q = q.clauses[0][0]
    assert isinstance(sub_q.clauses[1][0], TermQuery)
    assert sub_q.clauses[0][1] == "SHOULD"
    assert isinstance(sub_q.clauses[1][0], TermQuery)
    assert sub_q.clauses[1][1] == "SHOULD"
