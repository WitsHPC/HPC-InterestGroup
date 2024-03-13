import random
from timeit import default_timer as tmr
from typing import Dict, List

def get_dictionary(queries: List[str], names_to_search_in: List[str]) -> Dict[str, int]:
    return {n: i for i, n in enumerate(names_to_search_in)}

def v2_precompute(queries: List[str], names_to_search_in: List[str]) -> List[int]:
    # Precompute a dictionary that maps a name to its index. This is linear in the size of `names_to_search_in`
    dictionary_of_value_to_index = get_dictionary(queries, names_to_search_in)
    list_of_answers = []
    # Now linearly loop through the queries, this is linear in the size of this list, as the dictionary.get is constant
    for q in queries:
        list_of_answers.append(dictionary_of_value_to_index.get(q, -1))
    return list_of_answers

def get_queries_names(N: int):
    random.seed(2)
    queries = list(map(str, range(N)))
    names = list(map(str, range(N)))
    random.shuffle(names)
    return queries, names

def main():
    # Significantly faster!
    N = 10_000
    queries, names = get_queries_names(N)
    s = tmr()
    v2_precompute(queries, names)
    e = tmr()
    print(f"V2 took {e-s:.2f}s")
    
if __name__ == '__main__':
    main()