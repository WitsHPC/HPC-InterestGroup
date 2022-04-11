# Python Parallel Programming
- [Python Parallel Programming](#python-parallel-programming)
	- [Introduction](#introduction)
	- [What is Parallelism?](#what-is-parallelism)
	- [Why is it useful?](#why-is-it-useful)
	- [Use Cases](#use-cases)
	- [How to use it?](#how-to-use-it)
	- [Running Example](#running-example)
	- [Example 1 - Bash](#example-1---bash)
	- [Example 1 - Bash](#example-1---bash-1)
	- [Example 2 - Multi-Processing](#example-2---multi-processing)
	- [Example 2 - Multi-Processing](#example-2---multi-processing-1)
	- [Example 3 - Ray](#example-3---ray)
	- [Example 3 - Ray](#example-3---ray-1)
	- [More Complex Code](#more-complex-code)
	- [Communication](#communication)
	- [Sources](#sources)
	- [Conclusion](#conclusion)

---

## Introduction
+ Today we'll go over parallel programming in Python
+ Will cover basic (but common) use cases.

---

## What is Parallelism?
Basically, it is writing code to take advantage of multiple compute-elements (e.g. threads, cores, nodes, etc.), usually to make the code run faster.

---

## Why is it useful?
+ Can run experiments faster -- get results in hours and not days
+ Can actually use the hardware at your disposal

---

## Use Cases
+ GPU Programming -> Massively Parallel
+ Hash Code / Entelect University Cup -> Can perform more runs to get a better score
+ Research -> Need to run over many seeds / hyperparameters

---

## How to use it?
+ Today we'll be covering parallelisation in Python
+ There are a few ways to do this, we will consider the following 3:
	+ By hand / `bash`
	+ `multiprocessing` library
	+ `ray` library
+ Python threading is a lie!
	+ Does work for e.g. waiting tasks, like I/O
	+ But not great for CPU intensive tasks
	+ Because of GIL

---

## Running Example
As an example, consider a research setting. 
+ To get an accurate estimate of our results, we usually run experiments multiple times, often the same code with a different random seed.
+ This is a simple example (but occurs frequently). 
+ All executions are *independent*
+ This is called **embarrassingly parallel**
+ Suppose we must run the code with seeds 1-5, like `python experiment.py <seed>`

---
## Example 1 - Bash
+ Theoretically, you could just open 5 terminals and run the commands individually?
	+ Does not scale well at all (e.g. not feasible for 100 seeds)
+ You can automate this in `bash`, however, using 2 key aspects
	+ `&` to run in the background: `command &` runs `command` in the background.
	+ `for` to loop over the seeds.
	+ Valid for any language.


---
## Example 1 - Bash
::: block <!-- element style="font-size: 3rem;" -->

```bash
for i in {1..5}; do
	echo "Running $i"
	python experiment.py $i &
done
wait; # wait for final result
``` 
:::


---

## Example 2 - Multi-Processing
+ We could also do this *within* Python itself, using the built in `multiprocessing` library.

---

## Example 2 - Multi-Processing
::: block <!-- element style="font-size: 2rem;" -->

```python
import multiprocessing
from time import sleep

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
``` 
:::

---

## Example 3 - [Ray](https://www.ray.io/)
[Ray](https://www.ray.io/) is a super useful parallelisation library for Python, and aims to be simple and effective.

---

## Example 3 - [Ray](https://www.ray.io/)
::: block <!-- element style="font-size: 2rem;" -->

```python
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
``` 
:::

---


## More Complex Code
+ For example, a genetic algorithm that does its fitness calculations in parallel. Still embarrassingly parallel, but slightly more involved
+ I'd recommend `ray` for this, as:
	+ Quite simple, minimal hassle.
	+ Can even scale to multiple nodes without significant code change!

---

## Communication
This is slightly outside the scope of this talk, but if you are interested, look at the [ray collective library](https://docs.ray.io/en/latest/ray-more-libs/ray-collective.html).
If you are using `C/C++`, then [MPI](https://www.open-mpi.org/) and [OpenMP](https://www.openmp.org/) are industry standards.

---

## Sources
- https://medium.com/ki-labs-engineering/busting-the-myth-around-multithreading-in-python-5c29653affd2
- https://www.ray.io/
- https://docs.python.org/3/library/multiprocessing.html


---

## Conclusion
+ So, you can relatively easily make use of multiple cores in Python code.
+ And there are many reasons you might want to.
+ Ray is pretty convenient.
+ Communicating processes adds in some complications.


&shy;<!-- .element: class="fragment" --> See code at https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/parallel_programming/03_python