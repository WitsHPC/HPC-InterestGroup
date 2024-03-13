# 2022-10-24-CUDA

---

## Intro
+ Today we'll be going over some GPU programming
+ Specifically, we will cover some more CUDA techniques, and go over a couple of examples

---

## Recap
+ Quick recap of our [previous CUDA talk](https://github.com/WitsHPC/HPC-InterestGroup/tree/main/talks/programming/parallel_programming/04_cuda)
	+ CPUs rely on a few strong cores to do work
	+ GPUS rely on lots of weak cores to do work.
	+ Hence, not all tasks are amenable to GPUs, although many are.
	+ If you can use a GPU, generally you should; performance can be significantly higher.
+ Performance Considerations when using GPUs:
	+ Memory transfer is one of the most important things
	+ Utilisation of the hardware is important too
	+ Proper breaking down of the task into small pieces.


---

## CUDA C
+ You can write CUDA code using a slightly modified version of `C`.
+ To compile and run cuda code, you do need an NVIDIA GPU.
+ You need to explicitly deal with transferring memory onto the GPU and back to the CPU's RAM.
+ Similarly to other parallel programming paradigms, in CUDA, all threads run the same code, but they have different variables (e.g. in OpenMP this is the thread ID).
+ CUDA is a shared-memory paradigm, where all threads on the same GPU can access the same memory
	+ This is slightly more complicated for multiple GPUs.

---

## How To

+ Generally, you need to define a **kernel** function
+ `__global__ void vector_add(float* a, float* b, float*c)`
+ In *host* (i.e. CPU) code, you can call it as follows:
	+ `vector_add<<<gridDim, blockDim>>>(a, b, c);`
+ Here, `gridDim` and `blockDim` can either be integers, or `dim3` structs.
+ `int`:
	+ This effectively says run this kernel with `gridDim` x `blockDim` threads, grouped into `gridDim` blocks, each block having `blockDim` threads.
+ `dim3`
	+ Effectively, provide a 2D or 3D tiling, i.e. we have `gridDim.x` blocks in the x direction, `gridDim.y` in the y direction, etc.

---
## Indices
+ You can access the current thread's "thread ID" using the following:
	+ `blockIdx.x`: Which block is this in (can also use `.y`, `.z`, etc.)
	+ `threadIdx.x`: The thread index inside the block
+ To find a global thread ID, you can use something like `blockIdx.x * blockDim.x + threadIdx.x` 

---

## Memory
+ You need to allocate memory in the host code, using `cudaMalloc`. The idea is to do something like this:
+ `float* d_a;` -> Create a pointer that will store the address (in device memory) of the cuda variables. `d_` indicates device, which is useful for keeping track.
+ `cudaMalloc(&d_a, num_bytes);` -> This allocates `num_bytes` bytes of data on the GPU, and modifies the pointer such that it contains the correct memory address.
+ `cudaMemcpy(d_a, source, num_bytes, cudaMemcpyHostToDevice);` -> Copies data from `source` (a pointer to some CPU data) to `d_a` (pointer to GPU data).
+ `cudaFree(d_a);`  -> Frees memory
---

## Device Code
+ Generally, you cannot call host functions from cuda or vice versa.
+ To define a function in device code, you can do:
	+ `_device__ float sub(float x, float y) { return x - y; }`
+ This can not be called from host code, just from inside a cuda kernel.

---

## General
+ CUDA is prone to breaking, so use the `checkCudaErrors` wrapper function pretty much on all CUDA calls to catch errors.
+ `cudaDeviceSynchronize()` -> CUDA calls are sometimes asynchronous. Run this to let CPU code wait for the GPU to finish its current kernel call.
---

## Vector Sum
+ Some problem that illustrates computing. Computing a vector sum `NUM_ITERS` times.
```
__global__ void vector_add(long n, float* a, float* b, float*c){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = 0;
        for (int j =0; j < NUM_ITERS; ++j) c[i] += a[i] + b[i];
    }
}
```


---

## Getting Memory
```
    float *d_a, *d_b, *d_c;
    float *a, *b, *c, *tmp;
    long N = 1024 * 1024 * 32;
    long nbytes = N * sizeof(float);
    a   = (float*)malloc(nbytes);
    b   = (float*)malloc(nbytes);
    c   = (float*)malloc(nbytes);
    tmp = (float*)malloc(nbytes);

    int num_threads_per_block = 256;
    assert (nbytes % num_threads_per_block == 0);

    random_fill(N, a);
    random_fill(N, b);


    // malloc memory
    checkCudaErrors(cudaMalloc(&d_a, nbytes));
    checkCudaErrors(cudaMalloc(&d_b, nbytes));
    checkCudaErrors(cudaMalloc(&d_c, nbytes));
    checkCudaErrors(cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b, nbytes, cudaMemcpyHostToDevice));
```
---

## Call the Kernel
```
vector_add<<<N/num_threads_per_block, num_threads_per_block>>>(N, d_a, d_b, d_c);
```

---

## Performance
+ Memory transfers are generally very slow in CUDA.
+ If you can keep your data on the GPU, do so.
	+ Even if that requires some more computation
+ On the GPU itself, there are numerous considerations. 
	+ Again, memory accesses can be quite slow.
	+ CUDA provides *shared memory*, which is pretty much cache, for the programmer to control. Using this can speed things up a lot.
	+ Also, if adjacent threads access adjacent addresses in global memory, it is generally faster.
	+ There are also other types of memory, constant and texture. They can help.

---

## General Tips
+ Use the NVIDIA Profiler to optimise your code
+ Double check your results are correct, bad stuff often happens.
+ Look at memory access and how to optimise them
+ Consider using some compiler commands (e.g. `--use_fast_math`) or pragmas (e.g. `#pragma unroll`)

---

## Resources
+ CUDA by Example book
+ https://developer.nvidia.com/blog/even-easier-introduction-cuda/
+ https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/
+ See [here](https://github.com/Michael-Beukman/SFML-Mandelbrot) for a slightly more in-depth example