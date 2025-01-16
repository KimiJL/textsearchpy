from abc import ABC, abstractmethod
from typing import List
import re


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass


# gensim simple_tokenize pattern
PAT_ALPHABETIC = re.compile(r"(((?![\d])\w)+)", re.UNICODE)


class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        tokens = [match.group() for match in PAT_ALPHABETIC.finditer(text)]
        return tokens


class NGramTokenizer(Tokenizer):
    def __init__(self, min_gram: int, max_gram: int):
        super().__init__()
        self.min_gram = min_gram
        self.max_gram = max_gram

    def _convert_text_to_ngrams(self, text: str):
        if len(text) <= self.min_gram:
            return [text]

        ngrams = []
        for i in range(len(text)):
            # max_gram + 1 because range is not inclusive to last number
            for j in range(self.min_gram, self.max_gram + 1):
                if i + j > len(text):
                    break
                ngrams.append(text[i : i + j])

        return ngrams

    def tokenize(self, text: str) -> List[str]:
        return self._convert_text_to_ngrams(text)
