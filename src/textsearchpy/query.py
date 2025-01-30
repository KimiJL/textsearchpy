from abc import ABC, abstractmethod
from enum import Enum
from typing import List
from pydantic import BaseModel


class Query(BaseModel, ABC):
    @abstractmethod
    def to_query_string(self) -> str:
        pass


# no plans for fields for now
class TermQuery(Query):
    term: str

    def to_query_string(self) -> str:
        return f"{self.term}"


class Clause(str, Enum):
    SHOULD = "SHOULD"
    MUST = "MUST"
    MUST_NOT = "MUST_NOT"


class BooleanClause(BaseModel):
    query: Query
    clause: Clause


class BooleanQuery(Query):
    clauses: List[BooleanClause]

    def to_query_string(self) -> str:
        if not self.clauses:
            return ""
        query_string = ""
        for clause in self.clauses:
            sub_query_string = clause.query.to_query_string()
            if isinstance(clause.query, BooleanQuery):
                sub_query_string = f"({sub_query_string})"

            if len(query_string) > 0:
                if clause.clause is Clause.MUST:
                    sub_query_string = f" AND {sub_query_string}"
                elif clause.clause is Clause.SHOULD:
                    sub_query_string = f" OR {sub_query_string}"
                elif clause.clause is Clause.MUST_NOT:
                    sub_query_string = f" NOT {sub_query_string}"
            query_string += sub_query_string

        return query_string


class PhraseQuery(Query):
    terms: List[str]
    distance: int
    ordered: bool = False

    # ordered state is not reflected
    def to_query_string(self) -> str:
        join_terms = " ".join(self.terms)
        query_string = f'"{join_terms}"'
        if self.distance > 0:
            query_string += f"~{str(self.distance)}"

        return query_string


class WildcardQuery(Query):
    term: str

    def to_query_string(self) -> str:
        return f"{self.term}"


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

    # wildcard query
    elif "?" in token or "*" in token:
        term = tokens.pop(0)
        q = WildcardQuery(term=term)
        return q

    # term query
    else:
        term = tokens.pop(0)
        q = TermQuery(term=term)
        return q
