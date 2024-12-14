from abc import ABC
from typing import List, Tuple
from pydantic import BaseModel

# for now support TermQuery, PhraseQuery, BooleanQuery


class Query(BaseModel, ABC):
    pass


# no plans for fields for now
class TermQuery(Query):
    term: str


# TODO the str for AND OR NOT should be a enum
# TODO think about if AND OR NOT should be separated to optimize for AND + OR avoidance (likely not necessary)
class BooleanQuery(Query):
    clauses: List[Tuple[Query, str]]
