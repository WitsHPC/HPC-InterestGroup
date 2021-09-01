# Intro to Parallel Programming
- [Intro to Parallel Programming](#intro-to-parallel-programming)
- [Get Started](#get-started)
	- [Compiling and Running](#compiling-and-running)
- [What](#what)
	- [How does it function](#how-does-it-function)
- [Why](#why)
- [Learning](#learning)
- [How](#how)
- [OpenMP](#openmp)
	- [Compiling OpenMP programs.](#compiling-openmp-programs)
	- [Directives.](#directives)
	- [Parallel Region](#parallel-region)
	- [How many threads?](#how-many-threads)
	- [OMP Functions](#omp-functions)
	- [Simple, non trivial example.](#simple-non-trivial-example)
	- [Race conditions](#race-conditions)
	- [Data scope](#data-scope)
	- [Parallel For](#parallel-for)
	- [Putting everything together.](#putting-everything-together)
- [Exercises](#exercises)
- [Sources](#sources)
# Get Started

You can visit [https://github.com/Michael-Beukman/HPC-InterestGroup](https://github.com/Michael-Beukman/HPC-InterestGroup), specifically [https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/parallel_programming/01_omp](https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/parallel_programming/01_omp) for the notes and code.

## Compiling and Running

You can compile this code by simply typing `make`, and you can run a specific file by running `./bin/filename`, e.g. `./bin/01_hello`

If you are on a mac, or you wish to use another compiler, then you can run the following instead of make:

`CXX=new_compiler make`, e.g. `CXX=g++-9 make`

For mac specifically, you either need a gnu compiler, or the llvm clang compiler.

To do that, you can install Homebrew ([https://brew.sh/](https://brew.sh/)), which is a package manager for mac, and after installing that you can type:

`brew install gcc@7` and use `CXX=g++-7 make`
Tested and working on linux and mac.

# What

What is parallel programming? Well, simply put, it is writing code that will run on multiple compute elements (e.g. cores, nodes, threads, etc.), usually with the purpose of improving the runtime of an algorithm.

## How does it function

OpenMP uses a very interesting method for its thread management, namely that of fork-join.

How this works is that the main process gets 'forked', which means that a new thread (or process) is created, which is a complete duplicate of the main process, with some slight differences:

- It either has the same memory space, in the case of threads
- Or it has a separate memory space in the case of processes.

One key part is that the **same** code that gets executed by each thread. To do useful things, the main idea is to split the task up and make decisions based on the thread id.

For example:

```cpp
int main(){
	// do parallel things
	if (thread_id == 0){
		printf("Only thread 0 executes this line\n");
	}

	if (thread_id >= 1){
		printf("Thread %d executes this line\n", thread_id);
	}

}
```

This graphic details roughly how this works. We start with a main process, which splits up into multiple threads, and then at the end they join together again.

![Image](img/img.png)

An illustration of the fork join model, the main process is split up and then these threads are merged again.

# Why

Why would you want to do this?
The main reason is performance: Can I run my experiment / simulation / program in 20 minutes instead of 2 days.

A few notable examples

- Basically all computation that is performed on a GPU is only fast **because** it is done massively in parallel → Video games, Machine learning.
- Most machines nowadays have many cores, and this can allow us to run experiments / software much faster.

# Learning

There are many ways to do parallel programming, with lots of different frameworks and languages, for many different purposes. 

We will be using C++ with OpenMP in this talk, but other common frameworks are:

- MPI → Run across multiple nodes (also has a python api)
- CUDA → For NVIDIA GPUs, usually coded in C/C++ like syntax.
- Ray for Python.
- And some more, like the C++ threads library.

The principles are often similar between these different methods, but the syntax and exact mechanics might differ.

# How

The main thing to think about when writing parallel programs is problem decomposition: **How can we break our problem up into small, (hopefully) independent parts, that we can run in parallel.**

Some things to be wary of.

1. Correctness: The parallel program should always return the same result as the serial one. You need to validate this. If it doesn't return the same result, then it's not useful.
2. Should the program even be parallelised?
    1. If the serial program can solve it in reasonable time, or you don't expect a large increase in performance from parallelisation (maybe because of dependence, communication, hardware limitations, etc), then it might not be worth it to parallelise this.

# OpenMP

Open **M**ulti**P**rocessing is a C / C++ / Fortran library that allows programmers to write relatively little extra code to facilitate running across multiple cores and threads.

OpenMP uses a shared memory model, so each thread shares one memory space, and some variables could be shared and accessible from all threads.

The main programming construct for OpenMP is `#pragma` statements.

Here we will go over a few examples.

### Compiling OpenMP programs.

To compile an OpenMP program, you can use the `-fopenmp` flag when using gcc.

For example, `g++ prog.cpp -o prog -fopenmp`.

To use OpenMP, include the `omp.h` header at the top of the file.

For linux (or WSL), the above should work out of the box. For mac it's more tricky. The default clang doesn't support it out of the box, so you need to either use gcc or the llvm version of clang.

For the code in the repository, you should be able to just run `make`, and it should compile.

### Directives.

The basic structure of an OpenMP directive is the following:

```cpp
#pragma omp <directive> <clause>(<param1>, <param2>, ..., <paramn>)
{
	// some code
}
```

The curly braces are optional when there is only one line in the region, but it is recommended to keep it in anyway.

The above consists of a few different parts:

1. `#pragma omp`: This basically just tells the compiler that some omp directive is coming
2. `<directive>`: This determines what exactly should happen
3. `<clause>`: This is a way to add in some extra information, for example about variable scope, etc.

### Parallel Region

The simplest directive you can write is simply `parallel`, which will run the code inside the block using multiple threads. As detailed above, each thread executes the exact same code.

For example, a full program could be:

```cpp
#include <omp.h>
#include <stdio.h>
int main(){
	#pragma omp parallel
	{
		printf("Hello World\n");
	}
	return 0;
}
```

### How many threads?

How many threads will the above run on? Well, there are a few ways to set this up, but the easiest is to use the environment variable `OMP_NUM_THREADS`.

Thus, if you run `make` and run the program (e.g. `./bin/hello`) then by default it will use the number of cores that your machine has, but you can use `export OMP_NUM_THREADS=X` to use X threads. For parallel programming in general, it's often recommended to use only as many threads as cores, or sometimes use as many threads as hyperthreads (e.g. $2 \times \text{n\_cores}$). Using any more often oversaturates the CPU and results in worse performance.

### OMP Functions

There are some useful functions that most omp programs use. These are:

- `omp_get_num_threads()` → Returns how many threads are active
- `omp_get_thread_num()` → Returns the id of the thread that calls it.
- `omp_get_num_procs()` → Number of available cores.

Use the first two **only inside a parallel region**, otherwise they might not be properly defined.

### Simple, non trivial example.

Here we sum up the thread ids of all the threads 100 times.

```cpp
#include <omp.h>
#include <stdio.h>
int main(){
    int my_total = 0;
	#pragma omp parallel
	{
        for (int i=0; i<100; ++i)
            my_total += omp_get_thread_num();
	}
    printf("The total sum = %d\n", my_total);
	return 0;
}
```

### Race conditions

The above actually doesn't always produce the correct result! This is called a race condition, and it is caused because multiple threads read and write to the same variable, i.e. memory address.

This can take place (for example) in the following way:
| Thread 1|Thread 2 |
| ----------- | ----------- |
| read 5 |  |
| add 1 | |
| | read 5 |
| write 6 | |
| | add 2 | 
| | write 7| 


Which then causes some data to be lost, because two different threads are reading and writing to the same memory.

You could alleviate this using a few different techniques:

- Make an array `arr` of size `num_threads`, and then each thread `i` adds to `arr[i]`, and then after the parallel region, you can add up the values in `arr` on one thread.
- You can also use `pragma omp critical` for blocks that should only be executed by one thread at a time or `#pragma omp atomic` for variable accesses that should happen atomically.

### Data scope

In OpenMP we can have different data 'scopes', similar to how you can use `{}` in C/C++ to define the scope of the variable.

The main ones in OpenMP are:

- Shared: All the threads can access this variable
- Private: Each thread has its own unique, separate copy of this variable.

You can explicitly define these, but by default, all variables declared outside the parallel region will be shared and all the variables declared inside the region will be private.

### Parallel For

You could of course use what we've learnt up to now to write a program, e.g. something like:

```cpp
#include <omp.h>
#include <vector>
#include <stdio.h>
#include <cmath>

int main(){
	std::vector<int> v;
	// make the vector
	for (int i=0; i< 100000; ++i){
		v.push_back(i);
	}
	int N = v.size();
	// shared variable
	std::vector<int> totals (omp_get_num_threads(), 0);
	#pragma omp parallel
{
		int num_threads = omp_get_num_threads();
	  int per_thread = ceil(N / num_threadsv);
		int my_thread_num = omp_get_thread_num();
	
		int my_start = per_thread * my_thread_num;
		int my_end = my_start + per_thread;
	
	  for (int i=my_start; i < my_end; ++i){
				totals[my_thread_num] += v[i];
		}
	}

	// outside region
	int total = 0;
	for (auto i: totals) total += i;
	
	printf("TOTAL = %d\n", total);
	return 0;
}
```

But, OpenMP provides a very useful pragma, the `parallel for`

How this works, is as follows:

```cpp
#pragma omp parallel for
for (int i=0; i<N; ++i){
			totals[omp_get_thread_num()] += v[i];
}
// outside
int total = 0;
for (auto i: totals) total += i;
printf("TOTAL = %d\n", total);
```

It then splits up the for loop into roughly equal sized parts and performs them in parallel.

You could even use the `reduction` clause

```cpp
int N = v.size();
int global_total = 0;
// reduction, i.e. accumulate the results from each thread.
#pragma omp parallel for reduction(+:global_total)
 for (int i=0; i < N; ++i){
     global_total += v[i];
 }
printf("TOTAL = %d\n", global_total);

```

### Putting everything together.

With the above you can now basically start writing some more complex parallel programs.

See the example in 04_array_sum for a more complete example, with timing and validation.

# Exercises

For some exercises, feel free to do the following:

- Implement a parallel dot product
- Implement a parallel vector addition
- Or do any other interesting task in parallel.

# Sources

Victor Eijkhout's OpenMP / MPI book: [https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-basics.html](https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-basics.html)

[https://www.openmp.org/](https://www.openmp.org/)
