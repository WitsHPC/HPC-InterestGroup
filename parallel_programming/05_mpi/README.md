# Parallel Programming Using MPI
## Intro
+ Today we'll be speaking about MPI (Message Passing Interface), a specification that describes how to send and receive messages between different machines.
+ MPI has implementations for C/C++, Fortran, Python, etc.
+ Most High-Performance Computing software packages that work across multiple nodes use MPI.
+ OpenMP is targeted towards shared-memory settings, i.e. on a single node, while MPI is more focused on distributed computing across machines.
	+ Although MPI code can run just fine on just one machine using all of its cores.
	+ A common paradigm is to combine MPI and OpenMP, with MPI handling the inter-node communication and OpenMP parallelising computation on each node.
+ There are multiple implementations of MPI, each slightly different, but all follow the same specification.

---
## Goal Of This Talk
+ The goal of this is not to show you everything about MPI, but give you an idea of how it works and some basic usage.
+ Show you some performance considerations.



---
## Get Started
+ Install MPICH:  `sudo apt install mpich`
+ MPI works using `ranks`; each rank can be thought of as a specific process.
+ MPI executes the same code for each process. This can still be useful, though, as each rank has a separate ID to differentiate it from the others.
+ You can send messages between different ranks to communicate.
+ compile using `mpicc file.c -o file.run`
+ Run using `mpirun -np <procs> file.run`
	+ e.g. `<procs> = 4` runs 4 processes.


---
## Get Started

Simple Program

```
#include "mpi.h"
#include <stdio.h>
int main(){
    MPI_Init(NULL, NULL);

    int num_ranks, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    printf("Hello World from rank %d out of %d\n", my_rank, num_ranks);

    MPI_Finalize();
}
```

---
## What does this do?
+ `#include "mpi.h"` -> Include the relevant header file for the library
+ `MPI_Init(NULL, NULL);` Initialise MPI, can also pass in `argc` and `argv`, or `NULL` 
+ `MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);` -> Get the number of total processes
+ `MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);` -> Get the rank of the current process
+ `MPI_Finalize();` -> Clean Up


---
## Slightly More Complex Example
+ Generally, in something like OpenMP, we use shared memory, so it is trivial to access e.g. the same array with different indices.
+ In MPI, we do not have direct access to other processes' memory, we must explicitly send and receive messages.
+ The next example will send a number to the next rank, which will increment it by one and send it again to the next one.
+ `int num`
+ `MPI_Send(&mynum, 1, MPI_INT, my_rank + 1, TAG, MPI_COMM_WORLD);`
	+ `&mynum` -> ptr to data to send
	+ `1` -> How many values
	+ `MPI_INT` -> Datatype to send
	+ `my_rank + 1` -> Where to send the data to
	+ `TAG` -> Some arbitrary tag to group sends/receives
+ `MPI_Recv(&mynum, 1, MPI_INT, num_ranks - 1, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);`
	+ Similar parameters, just `from` instead of `to`
	+ And a `status`, extra information for debugging/logging
+ These are **blocking**, so be careful around deadlocks.


---
## Slightly More Complex Example
```
#include "mpi.h"
#include <stdio.h>
const int TAG = 1;
int main(){
    MPI_Init(NULL, NULL);

    int num_ranks, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // each process sends to n+1
    int mynum;
    if (my_rank == 0){
        mynum = 0;
        MPI_Send(&mynum, 1, MPI_INT, my_rank + 1, TAG, MPI_COMM_WORLD);

        MPI_Recv(&mynum, 1, MPI_INT, num_ranks - 1, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d received number %d\n", my_rank, mynum);
    }else{
        MPI_Recv(&mynum, 1, MPI_INT, my_rank - 1, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d received number %d\n", my_rank, mynum);
        mynum += 1;

        MPI_Send(&mynum, 1, MPI_INT, (my_rank + 1) % num_ranks, TAG, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
```


---
## Useful Examples
+ A more complex example, sum up numbers from 0 to N
+ Use asynchronous send and receive.
+ See `3.c`

---
## Pitfalls
+ Since MPI needs to send messages, you need to be very aware of the amount of data you send, as this can drastically slow your application down
+ On a cluster with slow interconnect, often running on multiple nodes is slower than running on one.

---
## More information
+ This was quite a short talk, there are many more things to MPI
	+ Broadcasting
	+ Asynchronous sending/receiving
	+ Barriers
	+ Reductions
+ And very many ideas regarding performance optimisation.
+ Have a look at the resources listed next, or feel free to ask questions.
---
## Resources
+ https://www.bu.edu/tech/support/research/training-consulting/online-tutorials/mpi/example1-2/example1_3/
+ https://mpitutorial.com/tutorials/mpi-hello-world/
+ https://theartofhpc.com/pcse.html