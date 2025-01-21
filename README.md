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
