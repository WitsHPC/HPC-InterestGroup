# GPUs & CUDA
![](cuda.png)
- [GPUs & CUDA](#gpus--cuda)
    - [Introduction](#introduction)
    - [CPUs](#cpus)
    - [GPUs](#gpus)
    - [CPUs vs GPUs](#cpus-vs-gpus)
    - [Performance Considerations](#performance-considerations)
    - [How to Use](#how-to-use)
    - [Demo - PyTorch](#demo---pytorch)
    - [Demo - CUDA](#demo---cuda)
    - [Conclusion](#conclusion)
    - [End](#end)
### Introduction


&shy;<!-- .element: class="fragment" --> Today we'll look at different types of hardware, their differences and some real-world effects of this.


---


### CPUs



A single core is basically one compute unit.
Most CPUs have multiple cores and server CPUs have lots.

CPUs are general purpose, and good at doing one thing.
Generally have some super-fast, but small cache memory. Often per core, or per small group of cores.
And larger, but slower RAM


---

### GPUs


+ Massive number of cores
	+ Lots of threads
+ Less general purpose
+ Memory local to some threads
+ Global memory for entire GPU.
+ Very good at tasks that have small and recurring subtasks
	+ Matrix Multiplication
	+ Rendering Graphics
	+ Cryptocurrency Mining
	+ Machine Learning



---


### CPUs vs GPUs


+ CPU can do everything
+ GPU cannot easily do everything
+ CPU is slow when faced with lots of small tasks
	+ Fast when given one long task
+ GPU: The opposite
	+ Lots of small tasks -> 10, 100x faster than a CPU.
+ Most proper ML done on GPUs


---


### Performance Considerations


+ Memory transfers and accesses can tremendously slow things down.
+ Efficiently load and use memory


---


### How to Use


+ Can write CUDA C / C++ code
	+ Compile with `nvcc`
	+ Low-level, manage memory.
	+ Arbitrary Code
+ Or, use PyTorch / Tensorflow in Python
	+ Mainly math-based operations.
	+ Much easier to write
		+ Also works on the CPU if a GPU is not available


---


### Demo - PyTorch



```
ROWS   = 10_000
HIDDEN = 10_000
COLS   = 10_000
torch.manual_seed(1)

A = torch.rand((ROWS, HIDDEN))
B = torch.rand((HIDDEN, COLS))

C = A @ B # matrix multiplication
```

```
CPU        took 5.08s. 5.08s was spent computing and 0.0s was spent on moving data.
CUDA       took 2.43s. 0.49s was spent computing and 1.94s was spent on moving data.
```



---

### Demo - CUDA
```
unsigned char *do_cuda()
{
    unsigned char *data = 0;
    int num_bytes = sizeof(unsigned char) * DIM * DIM * 4;
    cudaMalloc((void **)&data, num_bytes);
    dim3 gridSize, blockSize;
    gridSize.x = DIM;
    gridSize.y = DIM;
    cuda_julia<<<gridSize, 1>>>(data);
    return data;
}

```

---


### Conclusion


+ CUDA is a different way of computing compared to traditonal CPUs
+ In some cases, it can be very beneficial
+ In others, not so much
+ You need an NVIDIA GPU to be able to use it
+ PyTorch provides a useful way to leverage CUDA without much effort.


---


### End


+ The CUDA example code skeleton was from the Wits HPC Honours Course
+ Also see [here](https://github.com/nvidia/cuda-samples) and [here](https://developer.nvidia.com/cuda-example) for some of the utilities.
+ Usage: `make run` to compile and run the cuda code
+ Usage: `python pytorch.py` to run the PyTorch code.
+ `julia.cu` has the main cuda code.
