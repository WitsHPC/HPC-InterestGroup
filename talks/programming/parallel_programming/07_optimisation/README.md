# General Optimisation

This talk will cover the [competition](https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/competitions/2023_cuda/problem) we ran in August 2023.

## First Attempt -- Serial C++ Code
First, we have some simple, C++ code that works.

We first define the code to calculate neighbours:
```cpp

/**
 * @brief How many neighbours does a specific cell have
 * 
 * @param x 
 * @param y 
 * @param cells 
 * @return int 
 */
int neighbours(int x, int y, std::vector<std::vector<int>> &cells) {
    int N = 0;
    int W = cells[0].size();
    int H = cells.size();
    // for all neighbours
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            // not the cell itself
            if (!i && !j) continue;
            int tx = x + i;
            int ty = y + j;
            // wrap
            tx = (tx + W) % W;
            ty = (ty + H) % H;
            // add if the cell is active
            if (tx >= 0 && tx < W && ty >= 0 && ty < H) {
                N += cells[ty][tx];
            }
        }
    }
    return N;
}
```

Then a helper function to print a grid
```cpp
void print(std::vector<std::vector<int>> &cells) {
    string s = "";
    for (int y = 0; y < cells.size(); ++y) {
        for (int x = 0; x < cells[y].size(); ++x) {
            if (cells[y][x])
                s += "#";
            else 
                s += ".";
        }
        s += "\n";
    }
    cout << s;
}
```


Next we do our input

```cpp
int main(int argc, char** argv) {

    int w, h, n, m, A, B, C;
    cin >> w >> h >> n >> m >> A >> B >> C;
    std::vector<std::vector<int>> cells(h, std::vector<int>(w, 0));
    // now read the grid
    for (int i = 0; i < h; ++i) {
        string s;
        cin >> s;
        for (int j = 0; j < w; ++j) {
            if (s[j] == '#') {
                cells[i][j] = 1;
            }
        }
    }
    std::vector<std::vector<int>> buffer = cells;
```


Then we start our loop
```cpp
    for (int step=0; step < n; ++step){
        int step_one_indexed = step + 1;

        // update the new array
        for (int y = 0; y < cells.size(); ++y) {
            for (int x = 0; x < cells[y].size(); ++x) {
                int N = neighbours(x, y, cells);
                int is_on = cells[y][x];
                int new_val = 0;
                if (is_on){
                    if (N < A || N > B) new_val = 0;
                    else new_val = 1;
                }else{
                    if (N == C) new_val = 1;
                }
                buffer[y][x] = new_val;
            }
        }
```

At the end of the loop we swap the pointers to each buffer
```cpp
        // Swap the vectors so that the updated one is drawn in the next frame.
        std::swap(buffer, cells);

        if (step == 0 || (step + 1) % m == 0 || step == n - 1) 
        {
            print(cells);
        }
    }
}
```


### Results
Let's compile it (see `makefile`)

```bash
g++ -std=c++11 serial.cpp -o bin/serial
```

And run it
```
time ./bin/serial < ../../../competitions/2023_cuda/problem/examples/1.in > test/1.out

diff test/1.out ../../../competitions/2023_cuda/problem/examples/1.out
```

Ok, this takes around `11s` for this small input file, and `97s` using all of the files in the same way as we did during the competition.

## Optimise this!
Let us make one very simple change (add `-O3` to the compile command)
```
g++ -O3 -std=c++11 serial.cpp -o bin/serial_O3
```

This now takes `3.77s` for the small input file, and `22.68s` for the large input file. One compile command makes the code 4 times faster!




## Next Optimisation
Let us write it in `C` instead. The functions look very similar, now we just have `int* cells` instead of `std::vector<std::vector<int>> cells` and we have to pass the width and height of the grid to the functions.

I also use `printf` instead of `cout` to print the grid and `scanf` to read the input.
The swapping is also slightly different:

```c
int * temp = buffer;
buffer = cells;
cells = temp;
```

We still compile with `-O3`.

This runs in around `2.16s` on the small input file and `12.99s` overall. Why is this almost twice as fast as the `C++` code? My guess is the input/output is faster in `C` than in `C++`.

## But we can optimise the C code too

How do we optimise this?

A few things. First, make wrapping not work always. I.e., replace

```c
            tx = (tx + W) % W;
            ty = (ty + H) % H;
```
with 
```c
            // // wrap
            if (tx == -1 || tx == W)
                tx = (tx + W) % W;
            if (ty == -1 || ty == H)
                ty = (ty + H) % H;

```

Similarly, this if is unnecessary once we wrap
```c
            if (tx >= 0 && tx < W && ty >= 0 && ty < H) {
                int idx = ty * W + tx;
                N += cells[idx];
            }
```

Next, add this to the top of the main function. This sets stdout to be unbuffered, so it only prints at the end, instead of during the program, which makes it a bit faster.

```c
    setvbuf(stdout, NULL, _IOFBF, 16384 * 16);
```

Also, use `bool*` instead of `int*` for everything (you must include `<stdbool.h>` for this to work).

How fast is this? `0.86s` on the small file and `8.28` in total.


## OpenMP
Now, let us take this optimised `C` code and add three lines

```c
#include <omp.h> // top of file
```

```c
omp_set_num_threads(4); // top of the main function
```
Then, replace 
```c
for (int y = 0; y < h; ++y) {
```
With
```c
#pragma omp parallel for
for (int y = 0; y < h; ++y) {
```

And compile like:
```bash
gcc -O3 -fopenmp openmp_optim.c -o bin/ompOptim
```

Cool, `0.74s` on the small input and `6.52` overall! Pretty good for a one line change. Of course, as you have more cores, this will likely improve further.

## CUDA

Let's see how we can port our `C` code to CUDA

First, add the header files
```c
#include "./common/helper_cuda.h"
#include <cuda_runtime.h>
```

And make `int neighbours(int x, int y, int W, int H, int* cells)` a device function, like so
```c
__device__ int neighbours(int x, int y, int W, int H, int* cells)
```

And write a kernel to do our loops. This is basically the code we had in `C`, with one main difference! The two loops over cells
```c
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
```
are gone, replaced by 
```c
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * w + x;
    if (x < w && y < h){
```

```c
__global__ void game_of_life(int* d_Cells, int* d_Buffer, int w, int h, int A, int B, int C, int internal_iters) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * w + x;
    if (x < w && y < h){
        int N = neighbours(x, y, w, h, d_Cells);
        int is_on = d_Cells[idx];
        int new_val = 0;
        if (is_on){
            if (N < A || N > B) new_val = 0;
            else new_val = 1;
        }else{
            if (N == C) new_val = 1;
        }
        d_Buffer[idx] = new_val;
    }
}

```

Next, in the main code, we need to do some setup.

Such as, allocating memory
```c
    int* d_Cells;
    int* d_Buffer;
    
    checkCudaErrors(cudaMalloc(&d_Cells, w * h * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Buffer, w * h * sizeof(int)));
    
```

Copying it from the CPU to GPU
```c
    // copy data to d_Cells and d_Buffer;
    checkCudaErrors(cudaMemcpy(d_Cells, cells, w * h * sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_Buffer, d_Cells, w * h * sizeof(int), cudaMemcpyDeviceToDevice));
```

Then, in our main loop (over steps), we now have the following code
```c
        dim3 block_size(32, 32);
        dim3 num_blocks((w + block_size.x - 1) / block_size.x, (h + block_size.y - 1) / block_size.y);
        game_of_life<<<num_blocks, block_size>>>(d_Cells, d_Buffer, w, h, A, B, C, m);

        int * temp = d_Buffer;
        d_Buffer = d_Cells;
        d_Cells = temp;

        if (step == 0 || (step + 1) % m == 0 || step == n - 1) {
            // copy d_Cells to cells:
            checkCudaErrors(cudaMemcpy(cells, d_Cells, w * h * sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaDeviceSynchronize());
            print(w, h, cells);
        }
```

And at the end we have 
```c

    cudaFree(d_Buffer);
    cudaFree(d_Cells);
```

How does this perform? `1.55s` on the small input and `8.00s` overall. In particular, compared to optimised `C` code, cuda is `3x` slower on example 1, `13x` faster on example 2, about the same on examples 3 and 4, and twice as fast on example 5. 

This demonstrates that CUDA is not always better. For instance, if we have small problems, frequent outputs, etc.

## Conclusion
There are a few minor optimisations we can make to the CUDA code, but e.g., something like shared memory does not seem to improve performance much. Furthermore, on a large grid, `512x512` and 1M iterations, the OpenMP code takes ~8 minutes. The CUDA code takes `30s`. So, it can make a massive difference given the right problem.

Also, in a competition like this, it is often a good strategy to get something very unoptimised working first. Then, after that is confirmed, start optimisation. Trying to immediately go in and write an optimised version is harder, and the pressure is on to get it working.