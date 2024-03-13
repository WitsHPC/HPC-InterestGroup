from time import sleep
import multiprocessing

def main(seed: int):
	sleep(5)
	return seed ** 2


args = [0, 1, 2, 3, 4] # arguments to use
func = main            # function to run
N_WORKERS = multiprocessing.cpu_count() # How many cores
# Make a Pool and run all arguments using that.
with multiprocessing.Pool(N_WORKERS) as pool:
	results = pool.map(main, iterable=args)
	print("Answers = ", list(results))