#include <iostream>
#include <chrono>
#include <random>
#include <fstream>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " n" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);

    if (n < 0)
    {
        std::cerr << "Error: n must be non-negative." << std::endl;
        return 1;
    }

    int size = 1 << n;
    int *arr = new int[size];

    std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> distribution(0, 100);

    for (int i = 0; i < size; i++)
    {
        arr[i] = distribution(generator);
    }

    std::ofstream output_file("input.bin", std::ios::binary | std::ios::out);
    output_file.write(reinterpret_cast<const char *>(arr), size * sizeof(int));

    delete[] arr;

    return 0;
}
