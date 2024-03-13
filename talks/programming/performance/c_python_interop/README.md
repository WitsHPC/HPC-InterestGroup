# 2022-09-05-C-Python-Interop

---
## Intro
+ Today we'll be looking at how to call `C/C++` code in Python
+ Why you would want to
+ Some tips
+ And some examples

---
## Why
+ First of all, why would you want to do this?
+ Well, as we have seen before, C is generally significantly faster than Python at a lot of numerical computation
+ But, Python is often easier to work with.
+ This technique provides the best of both worlds, where the non-bottleneck parts can be done in Python, but the performance critical section can be moved into C
+ Or, you could use an existing C library in Python
	+ For instance easy parallelism using OpenMP
	+ Or even GPU compute using CUDA
+ But there are some things you need to be careful of

---

## How to do this?
First, create a standard C function, e.g. in `dot_product.c`:
```c
extern float dot_product(int length, float* a, float* b, float* ans){
    *ans = 0;
    for (int i=0; i < length; ++i){
        *ans += a[i] * b[i];
    }
}
```

+ And compile it like so:
+ `gcc dot_product.c  -shared -o dot_product.so`
+ This compiles it to a shared library

---

## How to do this - Python

+ Then, in Python, do the following:
+ `import ctypes`
+ Load the library
	+ `dot = ctypes.cdll.LoadLibrary('./dot_product.so')`
+ (Optional) Set the argument types: `dot.dot_product.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]`
+ Create memory buffers for the C code (`lista` and `listb` are just standard Python lists)
	+ `bufans = (ctypes.c_float*1)(0)`
	+ `bufa, bufb = (ctypes.c_float*N)(*lista), (ctypes.c_float*N)(*listb)`
+ Call it
	+ `dot.dot_product(len(lista), bufa, bufb, bufans)`
+ Get the answer: `print(bufans[0])`

---

## Performance
+ Let's dot product vectors of length 1M (`lista, listb`)
+ C code: `dot.dot_product(N, (f32*N)(*lista), (f32*N)(*listb), bufans)`
	+ **271 ms** ± 14.3 ms
+ Python: `ans = sum([a*b for a, b in zip(lista, listb)])`
	+ **87.2 ms** ± 5.19 ms
+ Yikes, why is Python 3.5 times faster than the C code?

---

## Data Transfer
+ Let's see how fast the buffer creation process is
+ `bufa, bufb = (f32*N)(*lista), (f32*N)(*listb)`
	+ **283 ms** ± 42 ms
+ Mmm... So that is quite slow.
+ Let's see how fast the dot product is when the data is already in the buffers
	+ `ans = dot.dot_product(N, bufa, bufb, bufans)`
	+ **1.25 ms** ± 28.4 µs
+ So, that is ~200 times faster than what it was!
+ Compare against numpy:
	+ `np.dot(lista,listb)`
	+ **200 ms** ± 24.8 ms per loop
+ So, numpy also has massive data transfer penalties.
	+ If we pre-load: `arra, arrb = np.array(lista), np.array(listb)`
	+ Then, `np.dot(arra, arrb)` is much faster
	+ **551 µs** ± 18.1 µs

---
## Optimisation
+ You can consider a few things to optimise the code with
	+ Compile with `-O3`
		+ `gcc -O3 dot_product.c  -shared -o dot_product_faster.so`
	+ Using the Intel compilers
	+ Using better algorithms

---

## Intrinsics
- We can also use intrinsics, e.g. AVX256, to improve our performance further.

```c
#include <immintrin.h>
extern float dot_product(int length, float* a, float* b, float* ans){
    // assume length % 8 == 0
    *ans = 0;
    const int ones = 0xFFFFFFFF;
    __m256i x = _mm256_set1_epi8(0xff);
    for (int i=0; i < length; i += 8){
        __m256 packed_a = _mm256_load_ps (a + i);
        __m256 packed_b = _mm256_load_ps (b + i);
        __m256 temp_ans = _mm256_dp_ps(packed_a, packed_b, 0xFF);
        *ans += temp_ans[0] + temp_ans[7];
    }
}
```


+ Which is roughly twice as fast as the numpy implementation!

---

## Takeaways
+ You can write C code to be called from Python.
+ This can be significantly faster under the right conditions.
+ Data transfer will tank your performance, so minimise it if you can.
+ Easy way to include OpenMP, CUDA, instrinsics, C libraries, etc.

---

## Resources
- https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
- https://realpython.com/python-bindings-overview/
- https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX
- https://github.com/WitsHPC/HPC-InterestGroup/tree/main/talks/programming/parallel_programming/02_omp_vectorised