from typing import Dict, List, Optional, Union
from pydantic import BaseModel
import re
import uuid
from src.normalizers import TokenNormalizer, LowerCaseNormalizer
from src.query import BooleanQuery, Query, TermQuery


class Document(BaseModel):
    text: str
    # metadata
    id: Optional[int] = None
    _index_tokens: Optional[List[str]] = None


# gensim simple_tokenize pattern
PAT_ALPHABETIC = re.compile(r"(((?![\d])\w)+)", re.UNICODE)


def tokenize(text: str) -> List[str]:
    tokens = []
    for match in PAT_ALPHABETIC.finditer(text):
        tokens.append(match.group())

    return tokens


class Index:
    def __init__(self):
        self.token_normalizers: List[TokenNormalizer] = [LowerCaseNormalizer()]
        # {token: doc_id}
        self.inverted_index: Dict[str, List[str]] = {}
        self.documents: Dict[str, Document] = {}
        # {token: {doc_id: [token_index]}}
        self.positional_index: Dict[str, Dict[str, List[int]]] = {}

    def _add_to_index(self, doc: Document):
        if doc.id is None:
            raise ValueError("Document ID cannot be None")

        self.documents[doc.id] = doc

        if doc._index_tokens:
            for tok_i, tok in enumerate(doc._index_tokens):
                # add to inverted index
                if tok in self.inverted_index:
                    self.inverted_index[tok].append(doc.id)
                else:
                    self.inverted_index[tok] = [doc.id]

                # add to positional index
                if tok not in self.positional_index:
                    self.positional_index[tok] = {}

                if doc.id in self.positional_index[tok]:
                    self.positional_index[tok][doc.id].append(tok_i)
                else:
                    self.positional_index[tok][doc.id] = [tok_i]

    def _normalize_tokens(self, tokens: List[str]):
        if not self.token_normalizers:
            return tokens

        for normalizer in self.token_normalizers:
            tokens = normalizer.normalize(tokens)

        return tokens

    def append(self, docs: List[Union[str, Document]]):
        for doc in docs:
            if isinstance(doc, str):
                doc = Document(text=doc)

            tokens = tokenize(doc.text)
            tokens = self._normalize_tokens(tokens)
            doc_id = uuid.uuid4()
            doc.id = doc_id
            doc._index_tokens = tokens
            self._add_to_index(doc)

    def search(self, query: Union[Query, str]) -> List[Document]:
        # TODO expand to query parser in future
        if isinstance(query, str):
            query = TermQuery(term=query)

        doc_ids = self._eval_query(query)

        docs = [self.documents[d_id] for d_id in doc_ids]

        return docs

    def _eval_query(self, query: Query) -> List[str]:
        if isinstance(query, BooleanQuery):
            and_set = None
            or_set = set()
            not_set = set()

            for clause in query.clauses:
                query = clause[0]
                query_condition = clause[1]

                doc_ids = self._eval_query(query)

                if query_condition == "MUST":
                    if and_set is None:
                        and_set = set(doc_ids)
                    else:
                        and_set = and_set.intersection(set(doc_ids))
                elif query_condition == "SHOULD":
                    or_set.update(doc_ids)
                elif query_condition == "MUST_NOT":
                    not_set.update(doc_ids)

            # if ANDs exists ORs are ignored
            match_doc_ids = and_set if and_set else or_set
            return match_doc_ids - not_set

        elif isinstance(query, TermQuery):
            # running same normalization on the search term to ensure consistency
            query_tokens = self._normalize_tokens([query.term])
            # TODO revisit: if normalization removes the token, consider no match
            if len(query_tokens) == 0:
                return []

            query_term = query_tokens[0]

            return self.inverted_index.get(query_term, [])

        else:
            raise ValueError("Invalid Query type")
