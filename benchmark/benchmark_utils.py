from textsearchpy.index import Index
from textsearchpy.query import Query
from typing import List
import time
import psutil


def create_index_from_data(data: List[str]):
    start = time.time()
    index = Index()
    index.append(data)
    end = time.time()
    elapsed_time = end - start
    print("Indexing Execution time:", elapsed_time, "seconds")
    print(f"Total Documents in Index: {len(index)}")
    print(f"Token Index Size: {len(index.inverted_index)}")
    print(f"Total tokens: {str(index.total_tokens)}")

    return index


def print_memory_usage():
    process = psutil.Process()
    mem_usage = process.memory_info().rss / 1024**2
    print(f"Memory Usage: {mem_usage} MiB")


def evaluate_queries(index: Index, queries: List[Query]):
    runtimes = []

    print(f"---------evaluating {len(queries)} queries----------")

    for query in queries:
        start = time.time()
        docs = index.search(query)
        end = time.time()
        elapsed_time = end - start
        runtimes.append(elapsed_time)
        print(f"Found {len(docs)} Docs")

    print(f"Average query runtime: {sum(runtimes) / len(runtimes)}")
    print(f"Max query runtime: {max(runtimes)}")
    print(f"Min query runtime: {min(runtimes)}")
