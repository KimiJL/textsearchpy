from typing import Dict, List, Optional
from pydantic import BaseModel
import re
import uuid


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
        self.token_analyzers: List = None
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

    def append(self, docs: List[Document]):
        for doc in docs:
            tokens = tokenize(doc.text)
            # TODO add token analyzers here
            doc_id = uuid.uuid4()
            doc.id = doc_id
            doc._index_tokens = tokens
            self._add_to_index(doc)

    def search(self, query: str, top_k: int = None) -> List[Document]:
        # expand logic as we go
        if query not in self.inverted_index:
            return []

        doc_ids = self.inverted_index[query]

        if top_k:
            doc_ids = doc_ids[:top_k]

        docs = [self.documents[d_id] for d_id in doc_ids]

        return docs
