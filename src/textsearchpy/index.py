import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Union
from pydantic import BaseModel
import uuid
import os
import math
from queue import PriorityQueue

from .tokenizers import SimpleTokenizer, Tokenizer
from .normalizers import TokenNormalizer, LowerCaseNormalizer
from .query import (
    BooleanQuery,
    Clause,
    PhraseQuery,
    Query,
    TermQuery,
    WildcardQuery,
    parse_query,
)


class IndexSearchError(Exception):
    pass


class IndexingError(Exception):
    pass


class Document(BaseModel):
    text: str
    # metadata
    id: Optional[str] = None
    # document size post processing
    count: Optional[int] = None
    # query match score
    score: Optional[float] = None


class QueryResult(BaseModel):
    """
    internal results for evaluating queries used to track metadata
    """

    # set or list out of convenience, to avoid recreating objects
    doc_ids: Union[Set[str], List[str]] = []

    # track doc_id: match_score of each document
    match_score: Optional[Dict[str, float]] = None


class Index:
    def __init__(
        self,
        token_normalizers: List[TokenNormalizer] = [LowerCaseNormalizer()],
        tokenizer: Tokenizer = SimpleTokenizer(),
    ):
        self.token_normalizers: List[TokenNormalizer] = token_normalizers
        self.tokenizer: Tokenizer = tokenizer

        # {token: doc_id}
        self.inverted_index: Dict[str, List[str]] = {}
        self.documents: Dict[str, Document] = {}
        # {token: {doc_id: [token_index]}}
        self.positional_index: Dict[str, Dict[str, List[int]]] = {}

        # tracked to calculate bm25 score avg doc length
        self.total_tokens = 0

    def __len__(self):
        return len(self.documents)

    def _add_to_index(self, doc: Document, tokens: List[str]):
        if doc.id is None:
            raise ValueError("Document ID cannot be None")

        self.documents[doc.id] = doc

        if tokens:
            for tok_i, tok in enumerate(tokens):
                # add to inverted index, first check if doc id already has been recorded in positional index to avoid dupe ids
                # TODO think about if inverted_index is needed vs using positional index only
                if (
                    tok not in self.positional_index
                    or doc.id not in self.positional_index[tok]
                ):
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

            self.total_tokens += len(tokens)

    def _normalize_tokens(self, tokens: List[str]):
        if not self.token_normalizers:
            return tokens

        for normalizer in self.token_normalizers:
            tokens = normalizer.normalize(tokens)

        return tokens

    def text_to_index_tokens(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        tokens = self._normalize_tokens(tokens)
        return tokens

    def append(self, docs: List[Union[str, Document]]):
        for doc in docs:
            if isinstance(doc, str):
                doc = Document(text=doc)

            tokens = self.text_to_index_tokens(doc.text)
            doc.count = len(tokens)
            if doc.id is not None:
                if doc.id in self.documents:
                    raise IndexingError(
                        f"Attempting to add a Document with ID: {doc.id} already exists in index"
                    )
            else:
                doc_id = uuid.uuid4().hex
                doc.id = doc_id

            self._add_to_index(doc, tokens)

    def search(self, query: Union[Query, str]) -> List[Document]:
        if isinstance(query, str):
            query = parse_query(query)

        query_result = self._eval_query(query, score=False)
        doc_ids = query_result.doc_ids

        docs = [self.documents[d_id] for d_id in doc_ids]

        return docs

    def retrieve_top_n(
        self, query: Union[Query, str], n: Optional[int] = None
    ) -> List[Document]:
        if isinstance(query, str):
            query = parse_query(query)

        query_result = self._eval_query(query, score=True)
        queue = PriorityQueue()

        for doc_id, score in query_result.match_score.items():
            queue.put((score, doc_id))

            if n and queue.qsize() > n:
                queue.get()

        result = []
        while not queue.empty():
            score, doc_id = queue.get()
            doc = self.documents[doc_id]
            doc.score = score
            result.insert(0, doc)

        return result

    def delete(self, docs: List[Document] = None, ids: List[str] = None) -> int:
        if docs is None and ids is None:
            raise Exception("docs or ids required to delete from index")

        ids_to_delete = []
        if docs:
            ids_to_delete = ids_to_delete + [
                d.id for d in docs if d.id in self.documents
            ]

        if ids:
            ids_to_delete = ids_to_delete + [id for id in ids if id in self.documents]

        # this doesn't seem very performant, could revisit to take advantage of bulk operations
        for d_id in ids_to_delete:
            doc = self.documents[d_id]

            # parses doc.text to tokens to clean up index, the tokens are not saved due to memory cost
            tokens = self.text_to_index_tokens(doc.text)
            for tok in tokens:
                if tok in self.inverted_index:
                    # TODO should handle better?
                    try:
                        self.inverted_index[tok].remove(d_id)
                    except ValueError:
                        pass
                    if len(self.inverted_index[tok]) == 0:
                        del self.inverted_index[tok]
                if tok in self.positional_index and d_id in self.positional_index[tok]:
                    del self.positional_index[tok][d_id]
                    if len(self.positional_index[tok]) == 0:
                        del self.positional_index[tok]
            self.total_tokens -= len(tokens)

        # remove documents
        for d_id in ids_to_delete:
            del self.documents[d_id]

        return len(ids_to_delete)

    def save(self, path: str, mkdir: bool = True) -> bool:
        if not os.path.exists(path) and mkdir:
            Path(path).mkdir(parents=True, exist_ok=True)

        if not os.path.isdir(path):
            raise Exception(f"input path: {path} should be a folder")

        document_file_path = os.path.join(path, "docs.jsonl")
        index_file_path = os.path.join(path, "index.json")

        if os.path.exists(document_file_path):
            raise Exception(f"{document_file_path} already exists")
        if os.path.exists(index_file_path):
            raise Exception(f"{index_file_path} already exists")

        with open(document_file_path, "w") as doc_file:
            for d in self.documents.values():
                json.dump(d.model_dump(), doc_file)
                doc_file.write("\n")

        with open(index_file_path, "w") as index_file:
            file_body = {
                "token_normalizers": [
                    t.__class__.__name__ for t in self.token_normalizers
                ],
                "tokenizer": self.tokenizer.__class__.__name__,
                "inverted_index": self.inverted_index,
                "positional_index": self.positional_index,
            }
            json.dump(file_body, index_file)

        return True

    def load_from_file(self, path: str) -> bool:
        if not os.path.exists(path) or not os.path.isdir(path):
            raise Exception(f"{path} must be a existing folder")

        document_file_path = os.path.join(path, "docs.jsonl")
        index_file_path = os.path.join(path, "index.json")

        with open(index_file_path, "r") as f:
            # TODO may want to validate tokenizer + normalizer set up against file
            loaded_index = json.load(f)

        self.inverted_index = loaded_index["inverted_index"]
        self.positional_index = loaded_index["positional_index"]

        saved_docs = {}
        with open(document_file_path, "r") as f:
            for line in f:
                doc = Document.model_validate(json.loads(line))
                saved_docs[doc.id] = doc

        self.documents = saved_docs

        return True

    def _eval_query(self, query: Query, score: bool) -> QueryResult:
        if isinstance(query, BooleanQuery):
            and_set = None
            or_set = set()
            not_set = set()

            match_score = {}

            for clause in query.clauses:
                query = clause.query
                query_condition = clause.clause

                sub_query_result = self._eval_query(query, score)
                doc_ids = sub_query_result.doc_ids

                if query_condition == Clause.MUST:
                    if and_set is None:
                        and_set = set(doc_ids)
                    else:
                        and_set = and_set.intersection(set(doc_ids))

                    if score:
                        for doc_id in and_set:
                            match_score[doc_id] = (
                                match_score.get(doc_id, 0)
                                + sub_query_result.match_score[doc_id]
                            )

                elif query_condition == Clause.SHOULD:
                    or_set.update(doc_ids)

                    if score and and_set is None:
                        for (
                            d_id,
                            sub_q_match_score,
                        ) in sub_query_result.match_score.items():
                            match_score[d_id] = (
                                match_score.get(d_id, 0) + sub_q_match_score
                            )

                elif query_condition == Clause.MUST_NOT:
                    not_set.update(doc_ids)

            # if ANDs exists ORs are ignored
            match_doc_ids = and_set if and_set is not None else or_set
            match_doc_ids = match_doc_ids - not_set
            query_result = QueryResult(doc_ids=match_doc_ids, match_score=match_score)
            return query_result

        elif isinstance(query, TermQuery):
            # running same normalization on the search term to ensure consistency
            query_tokens = self._normalize_tokens([query.term])
            # TODO revisit: if normalization removes the token, consider no match
            if len(query_tokens) == 0:
                return QueryResult()

            query_term = query_tokens[0]

            doc_ids = self.inverted_index.get(query_term, [])
            query_result = QueryResult(doc_ids=doc_ids)
            if score:
                match_score = {}
                for doc_id in doc_ids:
                    term_freq = len(self.positional_index[query_term][doc_id])
                    match_freq = len(doc_ids)
                    token_len = self.documents[doc_id].count
                    match_score[doc_id] = self._bm_25_score(
                        term_freq, match_freq, token_len
                    )
                query_result.match_score = match_score

            return query_result

        elif isinstance(query, PhraseQuery):
            terms = self._normalize_tokens(query.terms)

            if len(terms) == 1:
                # if phrase query is normalized to 1 term, treat it like a TermQuery
                return self._eval_query(TermQuery(term=terms[0]), score)
            elif len(terms) == 0:
                return QueryResult()

            # +1 to mimic edit distance instead of word distance i.e. "word1 word2" should be edit distance of 0, but word distance of 1
            distance = query.distance + 1
            ordered = query.ordered

            postings = []
            for term in terms:
                if term not in self.positional_index:
                    return QueryResult()
                postings.append(self.positional_index[term])

            doc_ids = []
            match_score = {}
            if len(terms) == 2:
                p1 = postings[0]
                p2 = postings[1]
                doc_ids, match_score = self._positional_intersect(
                    p1, p2, distance, ordered, score
                )
            elif len(terms) > 2:
                doc_ids, match_score = self._multi_term_positional_intersect(
                    postings, distance, ordered, score
                )
            query_result = QueryResult(doc_ids=doc_ids, match_score=match_score)
            return query_result
        elif isinstance(query, WildcardQuery):
            if "?" not in query.term and "*" not in query.term:
                # wildcard search not needed when wildcard symbol not present
                return self._eval_query(TermQuery(term=query.term), score)

            pattern = query.term.replace("?", ".")
            pattern = pattern.replace("*", ".+")
            re_pattern = re.compile(pattern)
            doc_ids = set()
            match_score = None
            for tok in self.positional_index.keys():
                if re_pattern.fullmatch(tok):
                    sub_query_result = self._eval_query(TermQuery(term=tok), score)
                    doc_ids.update(sub_query_result.doc_ids)

                    if score:
                        if not match_score:
                            match_score = sub_query_result.match_score

                        else:
                            for (
                                d_id,
                                sub_q_match_score,
                            ) in sub_query_result.match_score.items():
                                match_score[d_id] = (
                                    match_score.get(d_id, 0) + sub_q_match_score
                                )
            query_result = QueryResult(doc_ids=doc_ids, match_score=match_score)
            return query_result
        else:
            raise ValueError("Invalid Query type")

    def _find_match_doc_ids(self, search_index, to_search_index):
        match_doc_ids = []
        for doc_id in search_index.keys():
            if doc_id in to_search_index:
                match_doc_ids.append(doc_id)
        return match_doc_ids

    def _positional_intersect(
        self, p1: Dict, p2: Dict, k: int, ordered: bool, score: bool
    ):
        result = set()

        # iterate through the rarer term to find matching documents
        if len(p1.keys()) >= len(p2.keys()):
            doc_ids = self._find_match_doc_ids(p2, p1)
        else:
            doc_ids = self._find_match_doc_ids(p1, p2)

        freq_map = {}
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

                # add in doc frequency matched, temp should be matched length
                if score and len(temp) > 0:
                    freq_map[doc_id] = freq_map.get(doc_id, 0) + len(temp)

        match_score = {}
        if score and freq_map:
            for doc_id, term_freq in freq_map.items():
                match_score[doc_id] = self._bm_25_score(
                    term_freq, len(result), self.documents[doc_id].count
                )

        # should revisit to clean up algo so maybe we don't need to construct set to list here
        # needed currently because matched doc_id can duplicate
        return list(result), match_score

    def _multi_term_match_doc_ids(self, postings: List[Dict]):
        # start from the smallest candidate list to reduce search time
        sorted_postings = sorted(postings, key=lambda x: len(x.keys()))
        candidate = list(sorted_postings[0].keys())

        for posting in sorted_postings[1:]:
            candidate = [c for c in candidate if c in posting]

        return candidate

    def _multi_term_positional_intersect(
        self, postings: List[Dict], k: int, ordered: bool, score: bool
    ):
        result_doc_ids = set()

        doc_ids = self._multi_term_match_doc_ids(postings)

        freq_map = {}
        for doc_id in doc_ids:
            positions1 = postings[0][doc_id]
            positions2 = postings[1][doc_id]

            for pp1 in positions1:
                ranges = []
                # initialize search ranges, similar to two term phrase query
                for pp2 in positions2:
                    if ordered and pp2 < pp1:
                        continue

                    dis = abs(pp1 - pp2)
                    if dis <= k and dis != 0:
                        ranges.append((min(pp1, pp2), max(pp1, pp2)))
                    elif pp2 > pp1:
                        break

                for index, postings_k in enumerate(postings[2:]):
                    temp = []
                    positions_k = postings_k[doc_id]

                    for r in ranges:
                        for pp_k in positions_k:
                            if ordered and pp_k < r[1]:
                                continue

                            low = min(r[0], pp_k)
                            high = max(r[1], pp_k)
                            # - 1 - index subtracts the word distance count for matching tokens
                            # this way converts word distance into 'edit distance'
                            dis = high - low - 1 - index
                            if dis <= k and dis != 0:
                                temp.append((min(r[0], pp_k), max(r[1], pp_k)))
                            elif pp_k > r[1]:
                                break
                    ranges = temp

                if len(ranges) > 0:
                    result_doc_ids.add(doc_id)

                # ranges should represent all matches for starting position1
                if score and len(ranges) > 0:
                    freq_map[doc_id] = freq_map.get(doc_id, 0) + len(ranges)

        match_score = {}
        if score and freq_map:
            for doc_id, term_freq in freq_map.items():
                match_score[doc_id] = self._bm_25_score(
                    term_freq, len(result_doc_ids), self.documents[doc_id].count
                )

        return list(result_doc_ids), match_score

    def _bm_25_score(self, term_freq: int, match_freq: int, token_len: int):
        # default following elastic search
        k1 = 1.2
        b = 0.75

        doc_n = len(self.documents)

        idf = math.log((doc_n - match_freq + 0.5) / (match_freq + 0.5) + 1)

        top_term = term_freq * (k1 + 1)
        bot_term = term_freq + k1 * (
            1 - b + b * token_len / (self.total_tokens / doc_n)
        )

        return idf * top_term / bot_term
