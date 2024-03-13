#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
const int TAG_A = 1;
const int TAG_C = 3;


int main()
{
    long N = 10000000;
    MPI_Init(NULL, NULL);
    int num_ranks, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int START = 1;
    int COUNT = N / (num_ranks + START - 1);

    if (my_rank == 0)
    {
        double *A = (double*) malloc(num_ranks * sizeof (double));
        double C = 0;
        double *allCs = (double *)malloc(num_ranks * sizeof(double));
        double CSerial = 0;
        MPI_Request *requests = malloc(num_ranks * sizeof(MPI_Request));
        double start_par = MPI_Wtime();
        int i = 0;
        // send the start numbers
        for (i = START; i < num_ranks + START - 1; ++i)
        {
            A[i] = i * COUNT;
            MPI_Isend(&A[i], 1, MPI_DOUBLE, i + 1 - START, TAG_A, MPI_COMM_WORLD, &requests[i]);
        }
        // calculate my section
        for (int i = 0; i < COUNT; ++i)
        {
            C += (float)i;
        }

        // wait for the sending to complete
        MPI_Waitall((i - 1), &requests[START], MPI_STATUSES_IGNORE);


        // and receive
        for (i = START; i < num_ranks + START - 1; ++i)
        {
            MPI_Irecv(&allCs[i], 1, MPI_DOUBLE, i + 1 - START, TAG_C, MPI_COMM_WORLD, &requests[i]);
        }

        MPI_Waitall((i - START), &requests[START], MPI_STATUSES_IGNORE);

        // add all of the sub answers together
        for (i = START; i < num_ranks + START - 1; ++i)
        {
            C += allCs[i];
        }
        double end_par = MPI_Wtime();

        // check serial to validate and compare times
        double start_ser = MPI_Wtime();
        for (float i = 0; i < N; ++i)
            CSerial += i;

        double end_ser = MPI_Wtime();
        // check correctness
        if (fabs(CSerial - C) > 1e-5)
        {
            printf("ERROR: %f != %f\n", CSerial, C);
            return 1;
        }
        printf("Par Time = %lf, Serial Time = %lf\n", end_par - start_par, end_ser - start_ser);
        free(A);
        free(allCs);
        free(requests);
    }
    else
    {
        double num_to_start = 0;
        double myC = 0;
        // receive the number to start
        MPI_Recv(&num_to_start, 1, MPI_DOUBLE, 0, TAG_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // calculate
        for (float i = num_to_start; i < COUNT + num_to_start; ++i)
        {
            myC += i;
        }
        
        // send back
        MPI_Send(&myC, 1, MPI_DOUBLE, 0, TAG_C, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}