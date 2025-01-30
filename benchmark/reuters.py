from benchmark_utils import create_index_from_data, print_memory_usage, evaluate_queries
from textsearchpy.query import (
    TermQuery,
    BooleanQuery,
    BooleanClause,
    PhraseQuery,
    WildcardQuery,
)
from os import listdir
from os.path import isfile, join
from pathlib import Path

# https://www.nltk.org/nltk_data/ The Reuters-21578 benchmark corpus
REUTERS_DATA_PATH = "/Users/kimili/Downloads/reuters/total"


def load_corpus():
    data = []
    total_size = 0
    for f in listdir(REUTERS_DATA_PATH):
        path = join(REUTERS_DATA_PATH, f)
        if isfile(path):
            with open(join(REUTERS_DATA_PATH, f), "r", encoding="latin-1") as file:
                data.append(file.read())

            total_size += Path(path).stat().st_size

    print(f"Raw data size: {total_size * 0.000001} MB")
    return data


def run():
    data = load_corpus()
    print(f"Reuters data loaded with {len(data)} documents")

    print_memory_usage()

    index = create_index_from_data(data)

    print_memory_usage()

    evaluate_queries(
        index=index,
        queries=[
            TermQuery(term="payout"),
            TermQuery(term="income"),
            TermQuery(term="inventory"),
            TermQuery(term="management"),
            TermQuery(term="total"),
            TermQuery(term="outcome"),
        ],
    )

    evaluate_queries(
        index=index,
        queries=[
            BooleanQuery(
                clauses=[
                    BooleanClause(query=TermQuery(term="payout"), clause="MUST"),
                    BooleanClause(query=TermQuery(term="income"), clause="MUST"),
                ]
            ),
            BooleanQuery(
                clauses=[
                    BooleanClause(query=TermQuery(term="management"), clause="SHOULD"),
                    BooleanClause(
                        query=TermQuery(term="acquisition"), clause="MUST_NOT"
                    ),
                ]
            ),
            BooleanQuery(
                clauses=[
                    BooleanClause(query=TermQuery(term="payout"), clause="SHOULD"),
                    BooleanClause(query=TermQuery(term="income"), clause="SHOULD"),
                ]
            ),
            BooleanQuery(
                clauses=[
                    BooleanClause(query=TermQuery(term="management"), clause="SHOULD"),
                    BooleanClause(query=TermQuery(term="income"), clause="SHOULD"),
                    BooleanClause(
                        query=TermQuery(term="acquisition"), clause="MUST_NOT"
                    ),
                ]
            ),
        ],
    )

    evaluate_queries(
        index=index,
        queries=[
            PhraseQuery(
                terms=["management", "acquisition"], distance=10, ordered=False
            ),
            PhraseQuery(terms=["agree", "total"], distance=10, ordered=False),
            PhraseQuery(terms=["payout", "income"], distance=10, ordered=False),
        ],
    )

    evaluate_queries(
        index=index, queries=[WildcardQuery(term="s?t"), WildcardQuery(term="in*e")]
    )


if __name__ == "__main__":
    run()
