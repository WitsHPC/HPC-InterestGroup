#include <stdio.h>
#include <stdlib.h>
#include "./common/book.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include "./common/helper_cuda.h"
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>
#include "utils.h"

__global__ void matrix_mult(float* a, float* b, float* c, int n){
    // Each thread calculates a single element of the c matrix.
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < n && j < n) {
		for (int k = 0; k < n; k++) {
			c[i * n + j] += a[i * n + k] * b[k * n + j];
		}
	}
}

void matrix_mult_cpu(float* a, float* b, float* c, int n){
    // The single CPU process does all of the operations
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            for (int k = 0; k < n; k++) {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

void init(float* matrix, int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++)
        {
            matrix[i*n + j] = (float) (rand() / RAND_MAX);
        }
    }
}

int main(){
    // matrix multiplication
    int n = 1024;
    float *a = (float*)malloc(n*n*sizeof(float));
    float *b = (float*)malloc(n*n*sizeof(float));
    float *c = (float*)calloc(n*n, sizeof(float)); // init with zero
    float *c_cpu = (float*)calloc(n*n, sizeof(float));

    init(a, n);
    init(b, n);

    float *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc((void**)&d_a, n*n*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_b, n*n*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_c, n*n*sizeof(float)));

    float start_cuda = get_wtime();
    checkCudaErrors(cudaMemcpy(d_a, a, n*n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b, n*n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_c, c, n*n*sizeof(float), cudaMemcpyHostToDevice));

    dim3 blocks(32, 32);
    dim3 threads(32, 32);
    matrix_mult<<<blocks, threads>>>(d_a, d_b, d_c, n);

    checkCudaErrors(cudaMemcpy(c, d_c, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    float end_cuda = get_wtime();

    float start_cpu = get_wtime();
    matrix_mult_cpu(a, b, c_cpu, n);
    float end_cpu = get_wtime();

    // Compare c and c_cpu
    for (int i = 0; i < n*n; i++){
        if (abs(c[i] - c_cpu[i]) > 1e-5){
            printf("Error: %f, %f\n", c[i], c_cpu[i]);
            return 1;
        }
    }
    printf("Correct!\n");

    printf("The CPU took %f mseconds to run and the GPU took %f mseconds.\n", end_cpu - start_cpu, end_cuda - start_cuda);

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    free(a);
    free(b);
    free(c);
    free(c_cpu);

    return 0;
}