from textsearchpy.index import Index
from textsearchpy.query import TermQuery, BooleanQuery, BooleanClause, PhraseQuery
from os import listdir
from os.path import isfile, join
import time

# https://www.nltk.org/nltk_data/ The Reuters-21578 benchmark corpus
REUTERS_DATA_PATH = ""


def load_corpus():
    data = []
    for f in listdir(REUTERS_DATA_PATH):
        if isfile(join(REUTERS_DATA_PATH, f)):
            with open(join(REUTERS_DATA_PATH, f), "r", encoding="latin-1") as file:
                data.append(file.read())

    return data


def run():
    data = load_corpus()
    print(f"Reuters data loaded with {len(data)} documents")

    start = time.time()
    index = Index()
    index.append(data)
    end = time.time()
    elapsed_time = end - start
    print("Indexing Execution time:", elapsed_time, "seconds")
    print(f"Total Documents in Index: {len(index)}")

    q = TermQuery(term="payout")
    start = time.time()
    docs = index.search(q)
    end = time.time()
    elapsed_time = end - start
    print("TermQuery 'payout' Execution time:", elapsed_time, "seconds")
    print(f"Total Documents Found {len(docs)}")

    q = BooleanQuery(
        clauses=[
            BooleanClause(query=TermQuery(term="payout"), clause="MUST"),
            BooleanClause(query=TermQuery(term="income"), clause="MUST"),
        ]
    )
    start = time.time()
    docs = index.search(q)
    end = time.time()
    elapsed_time = end - start
    print("BooleanQuery 'payout AND income' Execution time:", elapsed_time, "seconds")
    print(f"Total Documents Found {len(docs)}")

    q = PhraseQuery(terms=["management", "acquisition"], distance=2, ordered=False)
    start = time.time()
    docs = index.search(q)
    end = time.time()
    elapsed_time = end - start
    print(
        "PhraseQuery 'management acquisition'~2 Execution time:",
        elapsed_time,
        "seconds",
    )
    print(f"Total Documents Found {len(docs)}")


if __name__ == "__main__":
    run()
