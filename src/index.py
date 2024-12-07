from typing import Dict, List, Optional
from pydantic import BaseModel
import re
import uuid
from src.normalizers import TokenNormalizer, LowerCaseNormalizer


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
        self.inverted_index: Dict[str, List[str]] = {}
        self.documents: Dict[str:Document] = {}

    def _add_to_index(self, doc: Document):
        if doc.id is None:
            raise ValueError("Document ID cannot be None")

        self.documents[doc.id] = doc

        if doc._index_tokens:
            for tok in doc._index_tokens:
                if tok in self.inverted_index:
                    self.inverted_index[tok].append(doc.id)
                else:
                    self.inverted_index[tok] = [doc.id]

    def _normalize_tokens(self, tokens: List[str]):
        if not self.token_normalizers:
            return tokens

        for normalizer in self.token_normalizers:
            tokens = normalizer.normalize(tokens)

        return tokens

    def append(self, docs: List[Document]):
        for doc in docs:
            tokens = tokenize(doc.text)
            tokens = self._normalize_tokens(tokens)
            doc_id = uuid.uuid4()
            doc.id = doc_id
            doc._index_tokens = tokens
            self._add_to_index(doc)

    def search(self, query: str, top_k: int = None) -> List[Document]:
        # expand logic as we go
        if query not in self.inverted_index:
            return []

        # assume query is single token currently, and needs to run same normalization
        query = self._normalize_tokens([query])
        # if query token is removed, consider as no match
        if len(query) == 0:
            return []
        query = query[0]

        doc_ids = self.inverted_index[query]

        if top_k:
            doc_ids = doc_ids[:top_k]

        docs = [self.documents[d_id] for d_id in doc_ids]

        return docs
