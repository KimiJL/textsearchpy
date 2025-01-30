# TextSearchPy
TextSearchPy is a high performance, easy to use, in-memory text search engine library written in Python.

## Install

```sh
pip install textsearchpy
```

## Quickstart

Default Index is created with SimpleTokenizer and LowerCaseNormalizer

```python
from textsearchpy import Index, Document

# create index with default settings
index = Index()

# add to index
index.append(["The quick brown fox"])

# alternatively, create Document object and append
doc = Document(text="jumps over the lazy dog")
index.append([doc])

# search with query string, returns list of Document objects
print(index.search("fox"))
print(index.search("fox OR dog"))
print(index.search("fox AND dog"))
print(index.search("fox NOT quick"))

# shows how input text is tokenized & normalized
print(index.text_to_index_tokens("The quick brown fox"))
```

## Query Syntax

Query can be written in string format (shown in quickstart) or by creating different Query objects

```python

query = "fox"

query = "fox AND dog NOT quick"

query = "(fox OR dog) AND (quick OR lazy)"

# proximity search with distance of 1
query = '"quick fox"~1'

```

### Supported Query Types:

**TermQuery - query with single token**

```python
from textsearchpy.query import TermQuery

query = "fox"
query = TermQuery(term="fox)
```


**BooleanQuery - sub queries with boolean condition**

Clause - "SHOULD", "MUST", "MUST_NOT"

BooleanClause accept any query type, thus it is possible to create a tree of grouped boolean queries

```python
from textsearchpy.query import TermQuery, BooleanClause, BooleanQuery

query = "(fox OR dog) AND (quick NOT lazy)"

term_1 = TermQuery(term="fox")
term_2 = TermQuery(term="dog")
query_left = BooleanQuery(
    clauses=[
        BooleanClause(query=term_1, clause="SHOULD"),
        BooleanClause(query=term_2, clause="SHOULD"),
    ]
)

term_3 = TermQuery(term="quick")
term_4 = TermQuery(term="lazy")
query_right = BooleanQuery(
    clauses=[
        BooleanClause(query=term_3, clause="SHOULD"),
        BooleanClause(query=term_4, clause="MUST_NOT"),
    ]
)

final_query = BooleanQuery(
    clauses=[
        BooleanClause(query=query_left, clause="MUST"),
        BooleanClause(query=query_right, clause="MUST"),
    ]
)
```


**PhraseQuery - multi term query with option to set proximity distance**

by default, terms in phrase is not order sensitive

the distance represents a term based edit distance, i.e. how many other terms are allowed within the boundry of the search terms

```python
from textsearchpy.query import PhraseQuery

query = '"brown fox"'
query = PhraseQuery(terms=["brown", "fox"], distance=0)
# to enforce order sensitivity
query = PhraseQuery(terms=["brown", "fox"], distance=0, ordered=True)


query = '"jumps dog"~3'
query = PhraseQuery(terms=["jumps", "dog"], distance=3)
# to enforce order sensitivity
query = PhraseQuery(terms=["jumps", "dog"], distance=3, ordered=True)

# any two additional terms are allowed between these three terms for searching
query = '"jumps lazy dog"~2'
```


**WildcardQuery - query with wildcard character to match**
Note: current implementation scans through entire index for simplicity, which can be slow

```python
from textsearchpy.query import WildcardQuery

# allow a single wildcard character, i.e. brown brawn
query = "br?wn"
query = WildcardQuery(term="br?wn")

# allow many wildcard character to match, i.e. bun, barn, brown
query = "b*n"
query = WildcardQuery(term=term)

```

## Benchmark

see ./benchmark for more info

### Index Performance

index created with SimpleTokenizer and LowerCaseNormalizer

terms - searchable tokens after tokenization and normalization

| dataset  | # of docs | # of unique terms | # of terms total | index time | raw data size | memory use |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- |
| Reuters  | 10788   | 29188 | 1330266 | 2.7s | 8.84 MB | ~135 MiB |
| Gutenberg| 1000    | 424612 | 68132774 | 120s | 396.30 MB | ~3700 MiB |


### Query Performance

calculated as the average latency of few different preset queries 

| dataset  | TermQuery | BooleanQuery | PhraseQuery | WildcardQuery |
| -------- | ------- | ------- | ------- | ------- | 
| Reuters  | 0.08ms   | 0.11ms | 0.06ms | 8.46 ms |
| Gutenberg| 0.07ms    | 0.10ms | 55ms | TODO |
