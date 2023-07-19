# Parallelisation
> Jonathan Faller


>Why do we need to parallelise applications?

- This can be answered by asking - why do we work in teams?
- Additionally - massive datasets

---
## 2 levels
1. Message Passing Interface - MPI
	- inter-node(computer)
2. Multi threading - OpenMP
	- intra-node

> Note: inter-node is used conceptually here, but it is not always the case that mpi is used strictly between nodes

---
## MPI
- basically
	- splits the application into different processes -> **ranks**
	- **ranks** carry out their part of the job
	- once all the **ranks** have communicated that they are finished -> bundled up nicely and combined into your result
---
## MPI
- can be done in a variety of languages, e.g.
	- c
	- c++
	- fortran
	- python
---
## MPI
- communicator
	- connects processes 
	- gives each process an identifier and arranges in an ordered topology (**ranks**)
		- this is mainly for organising groups of processes
	- *MPI_COMM* commands control this
---
## MPI
- point-to-point
	- things like *MPI_send* talk between processes 
		- its important for processes to be able to talk -> e.g. may need result from one process in another process
---
## MPI
- collective
	- communication within process groups (made by communicator)
	- directives like *MPI_bcast* and *MPI_reduce* used for group wide communication
		- as opposed to point-to-point
---
## MPI
- derived data types
	- you need to define the type of data sent between processes (*MPI_INT*, etc.)
[resources](https://en.wikipedia.org/wiki/Message_Passing_Interface)
---
## OpenMP
- languages
	- c
	- c++
	- fortran
- multi threading
	- shared memory multiprocessing
	- primary thread forks into sub threads
---

