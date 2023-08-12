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

const int N_ELEMENTS = 1024 * 1024 * 10;

// define the add function in cuda:
__global__ void add(float* a, float* b, float* answer){
    // get the index of the thread:
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_ELEMENTS){
        answer[tid] = a[tid] + b[tid];
    }
}

void add_cpu(float* a, float* b, float* answer){
    for (int i=0; i < N_ELEMENTS; i++){
        answer[i] = a[i] + b[i];
    }
}

int main(){

    float* data_1 = (float*)malloc(N_ELEMENTS * sizeof(float));
    float* data_2 = (float*)malloc(N_ELEMENTS * sizeof(float));
    float* answer = (float*)malloc(N_ELEMENTS * sizeof(float));
    srand(42);
    // fill up these elements randomly
    for (int i=0; i < N_ELEMENTS; i++){
        data_1[i] = ((float)rand() / (float) RAND_MAX);
        data_2[i] = ((float)rand() / (float) RAND_MAX);
    }

    // allocate memory on the device
    float *device_data_1, *device_data_2, *device_answer;
    checkCudaErrors(cudaMalloc(&device_data_1, N_ELEMENTS * sizeof(float)));
    checkCudaErrors(cudaMalloc(&device_data_2, N_ELEMENTS * sizeof(float)));
    checkCudaErrors(cudaMalloc(&device_answer, N_ELEMENTS * sizeof(float)));

    float start_time = get_wtime();
    
    // copy the memory to cuda:
    checkCudaErrors(cudaMemcpy(device_data_1, data_1, N_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_data_2, data_2, N_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaDeviceSynchronize());
    float end_time_memory = get_wtime();

    // call the function
    int block_size = 1024;
    int n_blocks = N_ELEMENTS / block_size + (N_ELEMENTS % block_size == 0 ? 0 : 1);

    float start_time_compute = get_wtime();
    add<<<n_blocks, 1024>>>(device_data_1, device_data_2, device_answer);

    checkCudaErrors(cudaDeviceSynchronize());
    float end_time_compute = get_wtime();
    
    float start_time_copy_back = get_wtime();

    // copy device_answer to the CPU:
    checkCudaErrors(cudaMemcpy(answer, device_answer, N_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));

    float end_time = get_wtime();


    float time_cuda_total   = end_time - start_time;
    float time_cuda_memory  = (end_time_memory - start_time) + (end_time - start_time_copy_back);
    float time_cuda_compute = end_time_compute - start_time_compute;
    // now calculate the sum to see if it was correct
    float sum = 0.0f;
    for (int i=0; i < N_ELEMENTS; i++){
        sum += answer[i];
    }

    // do this on the cpu. This is for two reasons:
    // 1. Compare speed on CPU vs GPU
    // 2. Ensure our GPU code is correct
    float* answer_cpu = (float*)malloc(N_ELEMENTS * sizeof(float));
    float start_time_cpu = get_wtime();
    add_cpu(data_1, data_2, answer_cpu);
    float end_time_cpu = get_wtime();
    float sum_cpu = 0.0f;
    for (int i=0; i < N_ELEMENTS; i++){
        sum_cpu += answer_cpu[i];
    }
    if (sum_cpu != sum){
        printf("The sum is not correct. The sum on the CPU is %f and the sum on the GPU is %f\n", sum_cpu, sum);
        exit(1);
    }
    printf("The computation took %f mseconds (%f memory transfer, %f compute) on the GPU and %f mseconds on the CPU\n", time_cuda_total, time_cuda_memory, time_cuda_compute,
            end_time_cpu - start_time_cpu);

    // free memory:
    free(data_1);
    free(data_2);
    free(answer);
    free(answer_cpu);
    checkCudaErrors(cudaFree(device_data_1));
    checkCudaErrors(cudaFree(device_data_2));
    checkCudaErrors(cudaFree(device_answer));
    return 0;
}