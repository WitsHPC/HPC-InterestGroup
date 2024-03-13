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