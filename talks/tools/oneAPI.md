# oneAPI

---

# What is oneAPI?

> [!info] A comprehensive set of libraries and tools of all things intel

It has:
- Intel CPU compilers (`c`, `c++`, `fortran`, etc.)
- Intel GPU (and technically other) compiler
- Intel MPI implementation
- Deep Neural Network (DNN) library
- And more
---

# Why Do We Care?

In HPC we like to make things go fast right…

Turns out some compilers are more equal than others, and when you use Intel's software with Intel's hardware…

---

# Lets See Some Real Results

---

# So what Will We Use?

| **Intel** | **Open source** |
| --------- | --------------- |
| icx       | gcc             |
| mkl       | openblas        |
| intelmpi  | openmpi         |

Otherwise: Same DAT file, same run command

---

# Results

- oneAPI $\to$ 1.9627e+02 GFLOPS
- Open source $\to$ 1.8532e+02 GFLOPS

Open source is catching up in recent years, but keep in mind this is unoptimised, and the optimisations intel provides are much better than that of open source

---

# Why?

As we mentioned previously, using Intel's software, with Intel's hardware has advantages.

They know exactly how to best optimise the software for the processor… they designed it.

Funnily enough, AMD hasn't got this right yet, and the Intel compilers are still better on AMD platforms (and getting the AMD compilers to work is a nightmare)

---

# SYCL

So why did I say earlier that oneAPI had a compiler for not only Intel GPUs?

`Sycl` can be used to compile for Intel, AMD, and NVIDIA GPUs, while you just write normal `c++`, once again showing Intel's unique position in software tooling.

---

# Training

Intel has made a range of training available, which can help you learn how to use the full range of tools

---

# Questions

---
