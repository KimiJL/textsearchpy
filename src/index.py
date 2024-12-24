from typing import Dict, List, Optional, Union
from pydantic import BaseModel
import re
import uuid
from src.normalizers import TokenNormalizer, LowerCaseNormalizer
from src.query import BooleanQuery, Clause, PhraseQuery, Query, TermQuery, parse_query


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

    def __len__(self):
        return len(self.documents)

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
        if isinstance(query, str):
            query = parse_query(query)

        doc_ids = self._eval_query(query)

        docs = [self.documents[d_id] for d_id in doc_ids]

        return docs

    def _eval_query(self, query: Query) -> List[str]:
        if isinstance(query, BooleanQuery):
            and_set = None
            or_set = set()
            not_set = set()

            for clause in query.clauses:
                query = clause.query
                query_condition = clause.clause

                doc_ids = self._eval_query(query)

                if query_condition == Clause.MUST:
                    if and_set is None:
                        and_set = set(doc_ids)
                    else:
                        and_set = and_set.intersection(set(doc_ids))
                elif query_condition == Clause.SHOULD:
                    or_set.update(doc_ids)
                elif query_condition == Clause.MUST_NOT:
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

        elif isinstance(query, PhraseQuery):
            # assume two terms for now
            t1 = query.terms[0]
            t2 = query.terms[1]
            # +1 to mimic edit distance instead of word distance i.e. "word1 word2" should be edit distance of 0, but word distance of 1
            distance = query.distance + 1
            ordered = query.ordered

            if t1 not in self.positional_index or t2 not in self.positional_index:
                return []

            p1 = self.positional_index[t1]
            p2 = self.positional_index[t2]
            doc_ids = self._positional_intersect(p1, p2, distance, ordered)
            return doc_ids
        else:
            raise ValueError("Invalid Query type")

    def _find_match_doc_ids(self, search_index, to_search_index):
        match_doc_ids = []
        for doc_id in search_index.keys():
            if doc_id in to_search_index:
                match_doc_ids.append(doc_id)
        return match_doc_ids

    def _positional_intersect(self, p1: Dict, p2: Dict, k: int, ordered: bool):
        result = set()

        # iterate through the rarer term to find matching documents
        if len(p1.keys()) >= len(p2.keys()):
            doc_ids = self._find_match_doc_ids(p2, p1)
        else:
            doc_ids = self._find_match_doc_ids(p1, p2)

        for doc_id in doc_ids:
            temp = []
            positions1 = p1[doc_id]
            positions2 = p2[doc_id]

            for pp1 in positions1:
                for pp2 in positions2:
                    # if phrase search is order sensitive, skip when term two position is before term one
                    if ordered and pp2 < pp1:
                        continue

                    dis = abs(pp1 - pp2)
                    # != 0 checks the token is not on the same position i.e. "word word" would match doc="word"
                    if dis <= k and dis != 0:
                        temp.append(pp2)
                    elif pp2 > pp1:
                        break

                # this is here to allow saving of positions, should revisit in the future
                # potentially could reduce extra processing if we don't need the matched token index
                while len(temp) > 0 and abs(temp[0] - pp1) > k:
                    temp.remove(temp[0])

                for ps in temp:
                    # for now just return doc_id for simplicity
                    # result.append((doc_id, pp1, ps))
                    result.add(doc_id)

        # should revisit to clean up algo so maybe we don't need to construct set to list here
        # needed currently because matched doc_id can duplicate
        return list(result)
