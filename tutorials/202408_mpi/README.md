# What is MPI?
>Lily de Melo

Message Passing Interface is a standardized message-passing system designed to facilitate communication between processes in parallel computing environments.

## Overview

The application is split into multiple processes - they are then assigned a rank and each process is given its own local memory.
- The rank is used to identify processes and can be used to manage communication between processes.
- Once each process has finished it's part of the job, they all bundle up into a result.

## How they communicate:

- **Communicator**: a communicator defines a group of processes that have the ability to communicate with one another - they use their ranks to do this.
- **Point-to-point**: sending and receiving messages between a pair of processes.
    - MPI_Send
    - MPI_Recv
- **Collective communication**: communication within a group.
    - MPI_Bcast - one process sends data to all the other processes.
    - MPI_Scatter - distributes distinct chunks of data from one process to all the others.
    - MPI_Gather - collect data from all other processes to one.
- **Synchronization:** synchronizes processes so that they reach a certain point of execution together.
    - MPI_Barrier - collective - all processes must meet the barrier before they proceed.
    - MPI_Wait - point-to-point - has to wait for a communication to finish.
- **Derived data types** - enable the definition of custom data structures that may be more complex than the basic built-in types.
    - In conclusion, this is too big brain and right now:
    
        ![dontcare.gif](https://github.com/froggomelo/test-repo/blob/e98dbd9ceb31a3bf0cc2d97c638a21b0c419962a/1000058015.gif)
    

In summary, communication is very important as one process may need results or information from another.

## Why MPI?

- In HPC, we are dealing with large datasets/problem sizes. That’s why we get all fast hardware so that we can speedup the computation.


>**Side note:** this is an opportunity to combine your software development skills with your knowledge of High-Performance Computing (HPC). Many scientific applications still need to be adapted for MPI or OpenMP. So I want to emphasize the importance of taking your math courses seriously. While you may not always need to master the mathematical concepts, having a solid understanding will be beneficial when it comes time to implement these concepts in practical applications. This is a just a snippet of a paper giving an overview of how an application works:
>
>![math](https://github.com/froggomelo/test-repo/blob/cc150fe9b357ed73b9f3cda98cbff62a96fd7b93/Untitled.png)  


# MPI vs OpenMP

Now what’s the difference between MPI and OpenMP? Let's start off with how they are similar, they are both parallelization techniques. 

## MPI

- **Distributed memory** - each process has its own local memory.
- **Message Passing** - processes communicate by sharing messages.

## OpenMP(Open Multi-Processing)

- **Shared memory** - all processes can access the same shared memory and shared variables.
- **Communication:** threads communicate implicitly by reading and writing shared variables in the common memory space.

## Contrasting

- OpenMP offload work into threads within a single process. MPI, on the other hand, can be used to distribute work across processes.
- OpenMP parallelized a single program through software-level threads, distributing threads across cores in the same node. MPI launches several instances of the same program (regardless of where) and allows you to partition work across these instances.
- OpenMP is most commonly used as directives to quickly parallelism structures like for-loops - that is to say, you can use a single line of code to ask OpenMP to parallelize a loop and it handles it for you.
    - MPI, on the other hand, is *much* more hands-on and low-level, requiring you to explicitly describe e.g. the communication between nodes, insert barriers to ensure nodes are broadly in sync, handle files carefully, etc.

## Ranks vs Threads Example

Follow the instructions in lammps_install.md to see the results of different combinations of threads and processes.

>**Considerations when running MPI-enabled programs:** Not all applications utilize every core or run optimally on every core. It is important to carefully consider the number of processes(and threads) you choose to use. Be sure to investigate the correlation between your problem size and the number of processes to ensure optimal performance.

# Example  
Follow the instructions here to run the example: [How to build and run an MPI program](https://github.com/WitsHPC/HPC-InterestGroup/blob/b54e9b4200d57a3ef85e7bc487b5eb9dfdd572f3/tutorials/202305_mpi/README.md)
```bash
#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Hello from process " << rank << " of " << size << std::endl;

    MPI_Finalize();

    return 0;
}
```

# Tutorial

You can try to implement the MPI send and receive by copying this code into a `.cpp` file and building and running it.

```bash
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  const int PING_PONG_LIMIT = 10;

  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // We are assuming 2 processes for this task
  if (world_size != 2) {
    fprintf(stderr, "World size must be two for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int ping_pong_count = 0;
  int partner_rank = (world_rank + 1) % 2;
  while (ping_pong_count < PING_PONG_LIMIT) {
    if (world_rank == ping_pong_count % 2) {
      // Increment the ping pong count before you send it
      ping_pong_count++;
      MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
      printf("%d sent and incremented ping_pong_count %d to %d\n",
             world_rank, ping_pong_count, partner_rank);
    } else {
      MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      printf("%d received ping_pong_count %d from %d\n",
             world_rank, ping_pong_count, partner_rank);
    }
  }
  MPI_Finalize();
}
```

- `int ping_pong_count = 0` - ping pong counter for limit
- `int partner_rank = (world_rank + 1) % 2` - calculates the partners rank - if we are on rank 0 it will result in 1 and vice versa(try out some math in your head)
- `while (ping_pong_count < PING_PONG_LIMIT)` - staying under our limit
- `if (world_rank == ping_pong_count % 2)` - makes sure we start at rank 0 and then we alternate
- `ping_pong_count++` - increase count
- `MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD)` - sends ping pong counter to its partner:
    - `&ping_pong_count`: Address of the data to send.
    - `1`: Number of elements to send (just one integer here).
    - `MPI_INT`: Data type of the elements.
    - `partner_rank`: Rank of the destination process.
    - `0`: Message tag (can be used to identify different types of messages).
    - `MPI_COMM_WORLD`: Communicator (all processes in this case).
- } else { - if it's not the current process's turn to send, it will be receiving
- `MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE)`:
    - `&ping_pong_count`: Address where the received data will be stored.
    - `1`: Number of elements to receive (just one integer here).
    - `MPI_INT`: Data type of the elements.
    - `partner_rank`: Rank of the source process.
    - `0`: Message tag.
    - `MPI_COMM_WORLD`: Communicator.
    - `MPI_STATUS_IGNORE`: Status object (can be used to get more info about the received message, ignored here).

# References

https://mpitutorial.com/tutorials/mpi-introduction/  
https://mpitutorial.com/tutorials/mpi-hello-world/  
https://arxiv.org/abs/2107.01243  
https://www.reddit.com/r/HPC/comments/qbxupv/new_to_hpc_my_basic_understanding_of_mpi_is_that/  
