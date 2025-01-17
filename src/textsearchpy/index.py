import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
import uuid
import os

from .tokenizers import SimpleTokenizer, Tokenizer
from .normalizers import TokenNormalizer, LowerCaseNormalizer
from .query import (
    BooleanQuery,
    Clause,
    PhraseQuery,
    Query,
    TermQuery,
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

        doc_ids = self._eval_query(query)

        docs = [self.documents[d_id] for d_id in doc_ids]

        return docs

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
            for tok in self.text_to_index_tokens(doc.text):
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
            terms = self._normalize_tokens(query.terms)

            if len(terms) == 1:
                # if phrase query is normalized to 1 term, treat it like a TermQuery
                return self.inverted_index.get(query_term, [])
            elif len(terms) == 0:
                return []

            # +1 to mimic edit distance instead of word distance i.e. "word1 word2" should be edit distance of 0, but word distance of 1
            distance = query.distance + 1
            ordered = query.ordered

            postings = []
            for term in terms:
                if term not in self.positional_index:
                    return []
                postings.append(self.positional_index[term])

            doc_ids = []
            if len(terms) == 2:
                p1 = postings[0]
                p2 = postings[1]
                doc_ids = self._positional_intersect(p1, p2, distance, ordered)
            elif len(terms) > 2:
                doc_ids = self._multi_term_positional_intersect(
                    postings, distance, ordered
                )

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

    def _multi_term_match_doc_ids(self, postings: List[Dict]):
        # start from the smallest candidate list to reduce search time
        sorted_postings = sorted(postings, key=lambda x: len(x.keys()))
        candidate = list(sorted_postings[0].keys())

        for posting in sorted_postings[1:]:
            candidate = [c for c in candidate if c in posting]

        return candidate

    def _multi_term_positional_intersect(
        self, postings: List[Dict], k: int, ordered: bool
    ):
        result_doc_ids = set()

        doc_ids = self._multi_term_match_doc_ids(postings)

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

        return list(result_doc_ids)
