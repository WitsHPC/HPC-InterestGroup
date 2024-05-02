#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

// #define N 1000000

void bitonic_sort(int *arr, int n, int up);
void bitonic_merge(int *arr, int n, int up);
int greatest_power_of_two_less_than(int n);
void compare_exchange(int *arr, int i, int j, int up);
void validateBitonic(const int *arr, int n, bool direction);

void validateBitonic(const int *arr, int n, bool direction)
{
    for (int i = 1; i < n; i++)
    {
        if (direction == (arr[i] < arr[i - 1]))
        {
            printf("INVALID BITONIC SEQUENCE\n");
            // Abort MPI execution
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    printf("VALID BITONIC SEQUENCE\n");
}

void bitonic_sort(int *arr, int n, int up)
{
    if (n == 1)
        return;

    int m = n / 2;
    bitonic_sort(arr, m, 1);
    bitonic_sort(arr + m, m, 0);
    bitonic_merge(arr, n, up);
}

void bitonic_merge(int *arr, int n, int up)
{
    if (n == 1)
        return;

    int m = greatest_power_of_two_less_than(n);
    for (int i = 0; i < n - m; i++)
        compare_exchange(arr, i, i + m, up);
    bitonic_merge(arr, m, up);
    bitonic_merge(arr + m, n - m, up);
}

int greatest_power_of_two_less_than(int n)
{
    int k = 1;
    while (k < n)
        k = k << 1;
    return k >> 1;
}

void compare_exchange(int *arr, int i, int j, int up)
{
    if ((arr[i] > arr[j]) == up)
    {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

int main(int argc, char **argv)
{

    vector<int> nums;

    int num;
    ifstream input_file("../gen/input.bin", ios::binary);

    while (input_file.read(reinterpret_cast<char *>(&num), sizeof(int)))
    {
        nums.push_back(num);
    }

    int N = nums.size();

    int rank, size;
    int arr[N], recv[N / 2];

    for (int i = 0; i < N; i++)
        arr[i] = nums[i];

    double start = MPI_Wtime();

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Bcast(arr, N, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = N / size;
    MPI_Scatter(arr, local_n, MPI_INT, recv, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    bitonic_sort(recv, local_n, 1);

    MPI_Gather(recv, local_n, MPI_INT, arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0)
    {
        bitonic_sort(arr, N, 1);
        // printf("Sorted array: ");
        // for (int i = 0; i < N; i++)
        //     printf("%d ", arr[i]);
        // printf("\n");
        printf("Time taken: %f seconds\n", (end - start) * 1000);
        // Save the time to a CSV file

        validateBitonic(arr, N, 1);

        ofstream output_file("output.csv", ios::app);
        output_file << (end - start) * 1000 << endl;
    }

    MPI_Finalize();
    return 0;
}
