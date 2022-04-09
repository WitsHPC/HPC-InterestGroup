from time import sleep
import ray # pip install ray

@ray.remote # Need this remote decorator
def main(seed: int):
    sleep(5)
    return seed ** 2

args = [0, 1, 2, 3, 4]
procs = [main.remote(arg) for arg in args] # func.remote(arg)

answers = ray.get(procs) # gets the answers
print("Answers = ", answers)

