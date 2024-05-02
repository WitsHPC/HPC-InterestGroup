#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <limits>
#include <omp.h>
#include <chrono>
#include <random>

using namespace std;

void validateBitonic(const vector<int> &arr, bool direction)
{
    int n = arr.size();

    for (int i = 1; i < n; i++)
    {
        if (direction == (arr[i] < arr[i - 1]))
        {
            printf("INVALID BITONIC SEQUENCE\n");
            exit(1);
        }
    }

    printf("VALID BITONIC SEQUENCE\n");
}

void checkIfPowerOfTwo(int n)
{
    // Check if the size of the input vector is a power of two
    if (n == 0 || (n & (n - 1)) != 0)
    {
        cout << "Not a power of two" << endl;
        exit(1);
    }
}

void bitonicMerge(vector<int> &arr, int low, int cnt, bool direction)
{
    if (cnt > 1)
    {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
        {
            if (direction == (arr[i] > arr[i + k]))
                swap(arr[i], arr[i + k]);
        }
        bitonicMerge(arr, low, k, direction);
        bitonicMerge(arr, low + k, k, direction);
    }
}

void bitonicSort(vector<int> &numbers, size_t left, size_t size)
{
    size_t groupSize = 2;
    auto itr = (size_t)log2((double)size);
    bool increasing;

    for (size_t i = 0; i < itr; ++i)
    {
#pragma omp parallel for private(increasing)
        for (size_t j = 0; j < size; j += groupSize)
        {
            increasing = ((j / groupSize) % 2 == 0);
            bitonicMerge(numbers, left + j, groupSize, increasing);
        }
        groupSize *= 2;
    }
}

int main()
{

    // vector<int> nums = {3, 7, 4};

    // omp_set_dynamic(0);
    int maxNumberOfThreads = omp_get_max_threads() - 2;
    omp_set_num_threads(maxNumberOfThreads);

    vector<int> nums;
    int num;
    ifstream input_file("../gen/input.bin", ios::binary);

    while (input_file.read(reinterpret_cast<char *>(&num), sizeof(int)))
    {
        nums.push_back(num);
    }

    int myVecSize = nums.size();

    bool isAscending = true; // true for ascending order, false for descending

    // checkIfPowerOfTwo(myVecSize);

    // Start the timer
    double start = omp_get_wtime();

    // bitonicSort(nums, 0, myVecSize, isAscending);

    bitonicSort(nums, 0, myVecSize);

    // Stop the timer
    double end = omp_get_wtime();

    // Print the time taken in ms
    cout << "Time taken: " << (end - start) * 1000 << " ms" << endl;

    // Save the time to a CSV file
    ofstream output_file("output.csv", ios::app);
    output_file << (end - start) * 1000 << endl;

    validateBitonic(nums, isAscending);

    return 0;
}
