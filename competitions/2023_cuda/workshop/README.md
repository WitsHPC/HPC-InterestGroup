# CUDA
## Introduction
CUDA allows us to program Graphics Processing Units (GPUs). CUDA is generally written as an extension to C/C++. 

GPUs have the following properties:
+ Massive number of cores and lots of threads
+ Less general purpose than CPUs
+ Memory local to some threads
+ Global memory for the entire GPU.
+ Very good at tasks that have small and recurring subtasks
    + Matrix Multiplication
    + Rendering Graphics
    + Cryptocurrency Mining
    + Machine Learning

Thus, for certain tasks --- where we have lots of small subproblems --- GPUs can be 10, 100+ times faster than a CPU. This tutorial will be a basic introduction to writing code for NVIDIA GPUs using CUDA.

## General CUDA Coding
A CUDA program is very similar to a standard C/C++ program. You start with a `main` function, and you can call other functions from there. The main difference is that you can define a function that will be run on the GPU. This is called a `kernel`. A kernel is defined as follows:
```c
__global__ void kernel_name(int *a, int *b, int *c) {
    // Do something
}
```

Note, the `__global__` is important, as it indicates that the function will be run on the GPU. The `kernel_name` is the name of the function. The arguments are the arguments to the function. Also notice that the arguments are pointers to memory on the GPU. This is because the GPU has its own memory, and we need to pass data to the GPU. We will see how to do this later.

Now, how do we call a kernel?
```c
kernel_name<<<num_blocks, num_threads>>>(a, b, c);
```

Here, `num_blocks` is the number of blocks of threads we want to run and `num_threads` is the number of threads per block. 

The number of threads per block is limited by the GPU hardware. For example, the NVIDIA Tesla K80 has a limit of 1024 threads per block. The number of blocks is limited by the GPU as well. For example, the NVIDIA Tesla K80 has a limit of 65535 blocks.

Now, how do we get data to the GPU? We need to allocate memory on the GPU. We do this as follows:
```c
int *d_a, *d_b, *d_c; // the d_ indicates that this is memory on the GPU, d is short for device
cudaMalloc((void **) &d_a, size * sizeof(int));
cudaMalloc((void **) &d_b, size * sizeof(int));
cudaMalloc((void **) &d_c, size * sizeof(int));
```

How do we get data onto the GPU then, once we have allocated the (empty) memory? We can use the function `cudaMemcpy` for this. The syntax of this function is `cudaMemcpy(destination, source, size, type)`. Size is the number of bytes to copy, and type is the type of copy. 
We will use `cudaMemcpyHostToDevice` to copy from the host (CPU) to the device (GPU). We will use `cudaMemcpyDeviceToHost` to copy from the device (GPU) to the host (CPU). Thus, to copy data to the GPU, we do the following:
```c
cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
```

Once we have allocated the memory, and copied the data from the host (i.e., CPU) to the device (i.e., GPU), we can now properly call the kernel:
```c
kernel_name<<<num_blocks, num_threads>>>(d_a, d_b, d_c);
```
In this case, the `d_c` pointer is effectively an output variable: the result of the kernel will be stored in `d_c`. We can now copy the data back to the host (i.e., CPU):
```c
cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
```

Finally, we must remember to free the memory when we are done with it:
```c
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```

### A concrete example
Now, the above has been somewhat abstract. Let's code a simple program that adds two vectors together.
First, let us create the main function that allocates memory, and writes random data into our two input vectors:
```c
int main() {
    int size = 1000;
    int *a, *b, *c;       // cpu memory
    int *d_a, *d_b, *d_c; // gpu memory

    // Allocate CPU Memory
    a = (int *) malloc(size * sizeof(int));
    b = (int *) malloc(size * sizeof(int));
    c = (int *) malloc(size * sizeof(int));

    // Allocate GPU Memory
    cudaMalloc((void **) &d_a, size * sizeof(int));
    cudaMalloc((void **) &d_b, size * sizeof(int));
    cudaMalloc((void **) &d_c, size * sizeof(int));

    // Write random data to a and b
    for (int i = 0; i < size; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }
}
```


Next, let us copy the data to the GPU (add this just below the previous code in the main function):
```c
cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
```

And let us call the kernel (still to be written) called `add_vectors`:
```c
add_vectors<<<1, size>>>(size, d_a, d_b, d_c);
```

Finally, let us copy the data back to the CPU, and free the memory:
```c
cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```
Let us print out the first 10 elements of `a`, `b` and `c`:
```c
for (int i = 0; i < 10; i++) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
}
```
And to close off, let us free the CPU memory:
```c
free(a);
free(b);
free(c);
```

Great. Now that the scaffold is done, let us write our kernel (called `add_vectors`, taking in the size `n`, three vectors, `a`, `b` and the output `c`):
```c
__global__ void add_vectors(int n, int *a, int *b, int *c) {
    int i = threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}
```

Ok, what is this `threadIdx.x` thing? Well, the GPU is made up of blocks of threads. Each thread has a unique ID. The `threadIdx.x` is the ID of the thread in the x direction. Thus, if we have a 1D block of threads, then `threadIdx.x` is the ID of the thread. If we have a 2D block of threads, then `threadIdx.x` is the ID of the thread in the x direction, and `threadIdx.y` is the ID of the thread in the y direction. If we have a 3D block of threads, `threadIdx.z` is the ID of the thread in the z-direction.


## Compiling
Now that we have some CUDA code, how do we compile it? Well, we need to use the `nvcc` compiler. This is the NVIDIA CUDA compiler. It is very similar to the `gcc` compiler. Let us compile our code:
```bash
nvcc -o add_vectors add_vectors.cu
```
And let us run it:
```bash
./add_vectors
```
## More Theoretical Ideas
### GPU Architecture
So, a GPU is very different to a CPU. Where in standard CPU programming, the default paradigm is single-threaded code, i.e., where only one thing happens at once, in GPU programming, the default paradigm is multi-threaded code, i.e., where many things happen at once. This is because GPUs have many cores, and thus can do many things at once. Concretely, GPUs are an example of the SIMD paradigm, i.e., Single Instruction Multiple Data. 
This means that the GPU has lots of threads. Each thread performs the same instruction, but on different data.

So, when we had our kernel above running `c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x]`, we were actually running this instruction on many different threads. Each thread had its own `threadIdx.x`, and thus each thread was adding a different element of the vectors.


When writing CUDA code it is also important to note that while there are multiple threads, they are organised into thread blocks, each thread block having multiple threads within it. Each thread block has a particular ID  (`blockIdx`) and each thread has a particular ID within this block (`threadIdx`).


So, if we had to make our above example more correct, we could instead write:
```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

### GPU Memory
GPUs have their own dedicated memory. Beyond that, the memory is divided into different hierarchies. Each of these trades off speed with capacity (e.g. either lots of memory at a relatively slower speed or less memory at a relatively faster speed). Although, there are also unique types of memory that are particularly good at certain operations The different types of memory are:
+ Global Memory   (this is the general memory that you allocate using `cudaMalloc`)
+ Shared Memory   (this is memory that is shared between threads in a block, and is very fast)
+ Constant Memory (this is memory that cannot change, i.e. it is read-only and provides fast read-times if all threads in a block access the same value)
+ Texture Memory (This is memory where caching works on a 2D grid, and is very fast for 2D access patterns).

We'll only be looking at global memory and shared memory in this tutorial. But feel free to investigate the other types of memory.


#### How to make a shared memory buffer
To make a shared memory buffer, we do the following *inside* a kernel:
```c
__shared__ int buffer[1024];
```
The `__shared__` indicates that it is shared memory. Further note that there is a limit to the amount of shared memory that can be allocated per block. For example, the NVIDIA Tesla K80 has a limit of 48KB of shared memory per block. The compiler will throw an error if you try to allocate more than this.

Now, how do we use this shared memory? After allocating it, we often copy some data from global memory into shared memory. We do this as follows:
```c
buffer[threadIdx.x] = a[threadIdx.x];
```

Now, if the shared memory is used by all threads going forward, it is helpful to have a way to say: do not go past this point in the code until all of the threads have reached it. We do this using the `__syncthreads()` function. This function will block until all threads in the block have reached it. Thus, we can use it to ensure that all threads have copied the data from global memory to shared memory before we continue. Thus, we can do the following:
```c
buffer[threadIdx.x] = a[threadIdx.x];
__syncthreads();

// continue with the rest of the kernel
```

As you can see from this example, if we need to copy the global memory into the shared memory. This involves reading the global memory. So, if you only need to read that memory once, it may not make sense to copy it into shared memory. However, if you need to read that memory multiple times, it can lead to massive speedups, as reading from shared memory is much faster than reading from global memory.

## More Advanced Concepts
### 2D and 3D Blocks

![Alt text](images/cuda_v1.png)
In this image, we have `gridDim.x = 2` and `blockDim.x = 4`. In each cell, the number indicates the value of `threadIdx.x`.


Now, we have been using a one-dimensional block of threads. This makes a lot of sense if our data structure is inherently one-dimensional, like a vector. However, when dealing with 2D or 3D data, such as images, it makes sense to use a 2D or 3D block of threads. This is because each thread can process a pixel in the image. Thus, we can have a 2D block of threads, where each thread processes a pixel in the image. This is very useful for image processing. In our kernel, we can effectively extend our indexing as follows:


```c
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

When calling the kernel, however, it is slightly more complex:

```c
dim3 num_blocks(32, 32);
dim3 num_threads(32, 32);
kernel_name<<<num_blocks, num_threads>>>(args, ...);
```

So here, we construct a `dim3` class for the number of blocks. By default, the values of `y` and `z` are 1.

### Device Code
+ Generally, you cannot call host functions from cuda or vice versa.
+ To define a function in device code, you can do:
	+ `_device__ float sub(float x, float y) { return x - y; }`
+ This can not be called from host code, just from inside a cuda kernel.

---


## Examples
Let us consider another example. To see the full code for the previous example, See `vector_addition.cu` for full code.
### Matrix Multiplication
See `matrix_mult.cu` for full code.


Now, suppose you have two matrices to multiply. To do so in a normal C++ program, you would do the following:
```c
void matrix_mult(float* a, float* b, float* c, int n){ // assumes c is zero initialised
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}
```


Now, a CUDA kernel to do the same thing would look like this. Here, each thread calculates one element of the output matrix `c`. Thus, we have `n * n` threads. Each thread calculates one element of `c`. 

```c
__global__ void matrix_mult(float* a, float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        for (int k = 0; k < n; k++) {
            c[i][j] += a[i][k] * b[k][j];
        }
    }
}
```


## Notes
### Checking Errors
The code I wrote here in the README is a bit simpler than what you would write normally. In particular, the following includes are important (note, this needs the `common` folder to work).
```c
#include "./common/book.h"
#include "./common/helper_cuda.h"
#include <cuda_runtime.h>
```

Secondly, we usually use the macro `checkCudaErrors` when running any CUDA function. This makes it easy to identify where any errors occur, as without calling this macro, it's harder to debug. The macro also terminates execution of the program when something fails.

This can be used as follows:
```c
checkCudaErrors(cudaMalloc(&device_data_1, N_ELEMENTS * sizeof(float)));
// instead of cudaMalloc(&device_data_1, N_ELEMENTS * sizeof(float))

checkCudaErrors(cudaMemcpy(device_data_1, data_1, N_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
//instead of cudaMemcpy(device_data_1, data_1, N_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice)
```


### Synchronise
`cudaDeviceSynchronize()` -> CUDA calls are sometimes asynchronous. Run this to let CPU code wait for the GPU to finish its current kernel call.

### Correct Block Sizes
Be sure to check that your block/grid sizes are correct. For instance, you must have the same number or more threads running that you have elements in your array. Similarly, you should also do bounds checking so you do not access memory that you shouldn't.
## Task
Write CUDA code to add four $n \times n$ matrices `a`, `b`, `c`, `d` elementwise into a variable `result`. This is a short tutorial problem. Feel free to use the other examples as skeletons.


## Optimisation
There are generally several ways to optimise your CUDA code. We won't be going over all of these in-depth, but consider looking into them:
1. Compiler optimisation flags. Similarly to other compilers, there are lots of flags you can add that change the behaviour of the compiler. Consider reading `nvcc --help`.
   1. Consider using some compiler commands (e.g. --use_fast_math) or pragmas (e.g. `#pragma unroll`)
2. Minimise memory transfers and communication between CPU and GPU. This can have a large effect. The CPU and GPU have different memory spaces. Thus, when you copy data from the CPU to the GPU, it takes time. Similarly, when you copy data from the GPU to the CPU, it takes time. Thus, if you can minimise the amount of data you copy between the CPU and GPU, you can speed up your code. This is why we often copy data from the CPU to the GPU, run many kernels on the GPU, and then copy the data back to the CPU. This is because copying data is slow, and running kernels on the GPU is fast.
3. If you can, minimise reading and writing to global memory, as it is comparatively slower than shared memory.
4. Consider reading into memory coalescing. By changing how you read/write memory, you can greatly improve speeds.
5. Consider optimising your block size and grid size; choosing the correct values could improve speed.
6. An obvious one is to minimise the amount of unnecessary work you do.


## Final Notes
This talk was particularly designed to not be comprehensive, and to just introduce you to CUDA. There is a lot more to cover, and I encourage you to explore CUDA more. I've found the best way to learn something is to use it for projects
## Resources
- [CUDA by example book](https://courses.ms.wits.ac.za/moodle/pluginfile.php/121464/mod_folder/content/0/cuda_by_example.book.pdf?forcedownload=1)
- [Simple Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [CUDA Docs/Tutorials](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/)