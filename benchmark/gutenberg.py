from benchmark_utils import create_index_from_data, print_memory_usage, evaluate_queries
from textsearchpy.query import TermQuery
from os import listdir
from os.path import isfile, join, isdir
import argparse
from pathlib import Path

# sample of project gutenberg books
DATA_PATH = ""


def load_corpus(n):
    data = []
    total_size = 0
    for dir in listdir(DATA_PATH):
        if isdir(join(DATA_PATH, dir)):
            for f in listdir(join(DATA_PATH, dir)):
                path = join(DATA_PATH, dir, f)
                if isfile(path):
                    with open(path, "r") as file:
                        data.append(file.read())

                    total_size += Path(path).stat().st_size

                    if len(data) == n:
                        print(f"Raw data size: {total_size * 0.000001} MB")
                        return data

    print(f"Raw data size: {total_size * 0.000001} MB")
    return data


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int)

    args = parser.parse_args()

    data = load_corpus(args.n)
    print(f"Gutenberg data loaded with {len(data)} documents")

    print_memory_usage()

    index = create_index_from_data(data)

    print_memory_usage()

    evaluate_queries(
        index=index,
        queries=[
            TermQuery(term="hearts"),
            TermQuery(term="child"),
            TermQuery(term="software"),
        ],
    )


if __name__ == "__main__":
    run()
