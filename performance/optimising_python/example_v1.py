import random
from timeit import default_timer as tmr
from typing import List

def v1_baseline_search_for_names_in_list(queries: List[str], names_to_search_in: List[str]) -> List[int]:
    """Takes in a list of queries and a list of names to search through.
        The function must return a list of integers, the indices of each element in the queries list, or -1 if that element is not found
    Args:
        queries (List[str]): 
        names_to_search_in (List[str]): 

    Returns:
        List[int]
    """
    list_of_answers = []
    for q in queries:
        for i, n in enumerate(names_to_search_in):
            if q == n:
                list_of_answers.append(i)
                break
        else:
            list_of_answers.append(-1)
    return list_of_answers

def get_queries_names(N: int):
    random.seed(2)
    queries = list(map(str, range(N)))
    names = list(map(str, range(N)))
    random.shuffle(names)
    return queries, names

def main():
    N = 10_000
    queries, names = get_queries_names(N)
    s = tmr()
    v1_baseline_search_for_names_in_list(queries, names)
    e = tmr()
    print(f"V1 took {e-s:.2f}s")
    
if __name__ == '__main__':
    main()