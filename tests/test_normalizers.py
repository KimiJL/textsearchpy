from src.normalizers import LowerCaseNormalizer

def test_lowercase_normalizer():
    normalizer = LowerCaseNormalizer()

    tokens = ["Hi", "JOHN", "what"]
    tokens = normalizer.normalize(tokens)

    assert tokens == ["hi", "john", "what"]