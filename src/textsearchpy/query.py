from abc import ABC
from enum import Enum
from typing import List
from pydantic import BaseModel


class Query(BaseModel, ABC):
    pass


# no plans for fields for now
class TermQuery(Query):
    term: str


class Clause(str, Enum):
    SHOULD = "SHOULD"
    MUST = "MUST"
    MUST_NOT = "MUST_NOT"


class BooleanClause(BaseModel):
    query: Query
    clause: Clause


class BooleanQuery(Query):
    clauses: List[BooleanClause]


class PhraseQuery(Query):
    terms: List[str]
    distance: int
    ordered: bool = False


class QueryParseError(Exception):
    pass


RESERVED_KEY_CHAR = set(["(", ")", '"'])


def tokenize_query(query: str) -> List[str]:
    tokens = []
    buffer = []

    for char in query:
        if char in RESERVED_KEY_CHAR:
            if len(buffer) > 0:
                tokens.append("".join(buffer))
                buffer.clear()
            tokens.append(char)
        elif char.isspace():
            if len(buffer) > 0:
                tokens.append("".join(buffer))
                buffer.clear()
        else:
            buffer.append(char)
    if len(buffer) > 0:
        tokens.append("".join(buffer))

    return tokens


def parse_query(query: str) -> Query:
    tokens = tokenize_query(query)
    return _parse_group_q(tokens)


def _parse_group_q(tokens: List[str]) -> Query:
    left = _parse_base_q(tokens)
    if len(tokens) == 0:
        return left

    query_clauses = []
    query_clauses.append(BooleanClause(query=left, clause=Clause.SHOULD))
    while tokens and tokens[0] != ")":
        op_type = None
        if tokens[0] in ["AND", "OR", "NOT"]:
            op_type = tokens.pop(0)
        else:
            # implied op is OR i.e. query = "word1 word2" = "word1 OR word2"
            op_type = "OR"

        right = _parse_base_q(tokens)

        if op_type == "OR":
            query_clauses.append(BooleanClause(query=right, clause=Clause.SHOULD))
        elif op_type == "NOT":
            query_clauses.append(BooleanClause(query=right, clause=Clause.MUST_NOT))
        elif op_type == "AND":
            # in cases of "word1 AND word2" word1 should also become MUST conditions
            if query_clauses[-1].clause == Clause.SHOULD:
                query_clauses[-1] = BooleanClause(
                    query=query_clauses[-1].query, clause=Clause.MUST
                )
            query_clauses.append(BooleanClause(query=right, clause=Clause.MUST))

    return BooleanQuery(clauses=query_clauses)


def _parse_base_q(tokens: List[str]):
    token = tokens[0]

    # group query
    if token == "(":
        tokens.pop(0)
        q = _parse_group_q(tokens)
        if len(tokens) == 0 or tokens[0] != ")":
            raise QueryParseError("Discovered Un-ending Bracket")
        tokens.pop(0)
        return q

    # phrase query
    elif token == '"':
        tokens.pop(0)
        if len(tokens) < 3:
            raise QueryParseError("Malformed Phrase Query Discovered")

        terms = []
        while tokens and tokens[0] != '"':
            terms.append(tokens.pop(0))

        if tokens[0] != '"':
            raise QueryParseError(
                "Phrase Query detected with unterminated double quote"
            )

        tokens.pop(0)
        distance = None
        if tokens and tokens[0].startswith("~"):
            distance_str = tokens.pop(0)[1:]
            try:
                distance = int(distance_str)
            except ValueError:
                raise QueryParseError(
                    f"Failed to parse phrase query edit distance: found {distance}"
                )

        distance = distance if distance else 0

        return PhraseQuery(terms=terms, distance=distance)

    # term query
    else:
        term = tokens.pop(0)
        q = TermQuery(term=term)
        return q
