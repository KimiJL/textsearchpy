from abc import ABC, abstractmethod
from typing import List


class TokenNormalizer(ABC):
    @abstractmethod
    def normalize(self, tokens: List[str]) -> List[str]:
        pass


class LowerCaseNormalizer(TokenNormalizer):
    def normalize(self, tokens: List[str]) -> List[str]:
        t_tokens = [t.lower() for t in tokens]
        return t_tokens


class StopwordsNormalizer(TokenNormalizer):
    def __init__(self, stopwords: List[str] = None):
        super().__init__()
        if stopwords is not None:
            if not isinstance(stopwords, list):
                raise Exception(
                    "StopwordsNormalizer custom stopwords must be a list of string"
                )

            self.stopwords = set(stopwords)
        else:
            self.stopwords = set(DEFAULT_STOP_WORDS)

    def normalize(self, tokens: List[str]) -> List[str]:
        if not self.stopwords:
            return tokens
        t_tokens = [t for t in tokens if t not in self.stopwords]
        return t_tokens


# from nltk english list, in lowercase only
DEFAULT_STOP_WORDS = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
]
