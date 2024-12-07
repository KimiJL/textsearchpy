from abc import ABC, abstractmethod
from typing import List


class TokenNormalizer(ABC):
    @abstractmethod
    def normalize(self, tokens: List[str]):
        pass
    
class LowerCaseNormalizer(TokenNormalizer):
    def normalize(self, tokens: List[str]):
        t_tokens = []
        for token in tokens:
            t_tokens.append(token.lower())
        return t_tokens