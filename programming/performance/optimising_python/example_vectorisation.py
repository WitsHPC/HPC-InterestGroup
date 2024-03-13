import numpy as np
from timeit import default_timer as tmr

def my_dot(a, b):
    ans = 0
    for i, j in zip(a, b): ans += i * j
    return ans

def main():
    N = 10000_000
    
    a = np.random.rand(N)
    b = np.random.rand(N)
    s = tmr()
    print("Dot =", my_dot(a, b))
    e = tmr()
    print(f"Loops took {e-s:.2f}s")
    
    s = tmr()
    print("Dot =", np.dot(a, b))
    e = tmr()
    print(f"np.dot took {e-s:.2f}s")
    pass
if __name__ == '__main__':
    main()