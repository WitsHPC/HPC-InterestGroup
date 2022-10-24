// The code skeleton was obtained from the wits HPC course, and I then implemented the necessary CUDA / OpenMP functions
// Also see here: https://docs.nvidia.com/cuda/cuda-samples/index.html and here: https://developer.nvidia.com/cuda-example
#include "./common/book.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include <assert.h>
#include "./common/helper_cuda.h"
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>

#define NUM_ITERS 20

void random_fill(long N, float* a){
    for (int i=0; i < N; ++i){
        a[i] = rand() / (float) RAND_MAX;
    }
}


__global__ void vector_add(long n, float* a, float* b, float*c){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = 0;
        for (int j =0; j < NUM_ITERS; ++j) c[i] += a[i] + b[i];
    }
}

float get_wtime(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return ((long)t.tv_sec * 1000000 + (long)t.tv_nsec / 1000)/1000.0f;

}

int main(void)
{
    float *d_a, *d_b, *d_c;
    float *a, *b, *c, *tmp;
    // get data
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
    float cuda_mem_start = get_wtime();
    checkCudaErrors(cudaMalloc(&d_a, nbytes));
    checkCudaErrors(cudaMalloc(&d_b, nbytes));
    checkCudaErrors(cudaMalloc(&d_c, nbytes));
    checkCudaErrors(cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b, nbytes, cudaMemcpyHostToDevice));
    float cuda_mem_end = get_wtime();
    
    // calculate
    float cuda_compute_start = get_wtime();
    vector_add<<<N/num_threads_per_block, num_threads_per_block>>>(N, d_a, d_b, d_c);
    checkCudaErrors(cudaDeviceSynchronize());
    float cuda_compute_end = get_wtime();

    // copy results back
    float cuda_cp_back_start = get_wtime();
    checkCudaErrors(cudaMemcpy(tmp, d_c, nbytes, cudaMemcpyDeviceToHost));
    float cuda_cp_back_end = get_wtime();

    // check correctness and speedup on the CPU.
    float cpu_compute_start = get_wtime();
    for (long i=0; i < N; ++i){
        c[i] = 0;
        for (int j=0; j < NUM_ITERS; ++j){
            c[i] += a[i] + b[i];
        }
    }
    float cpu_compute_end = get_wtime();

    for (long i=0; i < N; ++i){
        assert(c[i] == tmp[i]);
    }
    
    float cuda_mem = cuda_cp_back_end - cuda_cp_back_start + cuda_mem_end - cuda_mem_start;
    float cuda_compute = cuda_compute_end - cuda_compute_start;
    float cpu_compute = cpu_compute_end - cpu_compute_start;

    printf("CPU TIME %f | CUDA Time: %f | CUDA Compute Time: %f | CUDA Mem Time %f\n", cpu_compute, cuda_compute + cuda_mem, cuda_compute, cuda_mem);

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    free(a);
    free(b);
    free(c);
    free(tmp);
}

