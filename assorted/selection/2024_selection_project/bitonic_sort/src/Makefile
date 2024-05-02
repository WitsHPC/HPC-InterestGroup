#CHANGE COMPILER OPTIONS HERE
OMPFLAG = 
CC = g++
MPI_CC = 
CFLAGS =

# DON'T CHANGE THE CODE BELOW
all:
ifeq ($(strip $(MPI_CC) $(OMPFLAG)),)
	$(CC) $(CFLAGS) -fopenmp bitonic.cpp -o bitonic
else ifneq ($(strip $(MPI_CC)),)
	$(MPI_CC) $(CFLAGS) $(OMPFLAG) bitonic_mpi.cpp -o bitonic
else ifneq ($(strip $(OMPFLAG)),)
	$(CC) $(CFLAGS) $(OMPFLAG) bitonic_omp.cpp -o bitonic
else
	$(CC) $(CFLAGS) bitonic.cpp -o bitonic
endif

clean:
	rm -vf bitonic
