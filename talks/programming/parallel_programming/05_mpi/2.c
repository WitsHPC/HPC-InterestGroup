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

        // send first -- this is blocking, so rank 0 must send first
        MPI_Send(&mynum, 1, MPI_INT, my_rank + 1, TAG, MPI_COMM_WORLD);
        
        // then wait for the receive call.
        MPI_Recv(&mynum, 1, MPI_INT, num_ranks - 1, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d received number %d\n", my_rank, mynum);
    }else{
        // first receive
        MPI_Recv(&mynum, 1, MPI_INT, my_rank - 1, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d received number %d\n", my_rank, mynum);
        mynum += 1;
        // and then send
        MPI_Send(&mynum, 1, MPI_INT, (my_rank + 1) % num_ranks, TAG, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}