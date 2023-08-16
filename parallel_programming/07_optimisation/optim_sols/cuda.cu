#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include "./common/helper_cuda.h"
#include <cuda_runtime.h>

__device__ int neighbours(int x, int y, int W, int H, int* cells) {
    int N = 0;
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
                int idx = ty * W + tx;
                N += cells[idx];
            }
        }
    }
    return N;
}

void print(int W, int H, int* cells) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (cells[y * W + x])
                printf("#");
            else 
                printf(".");
        }
        printf("\n");
    }
}

// game of life kernel in CUDA:

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

int main(int argc, char** argv) {

    int w, h, n, m, A, B, C;
    scanf("%d %d %d %d %d %d %d\n", &w, &h, &n, &m, &A, &B, &C);
    int* cells  = (int*) malloc(w * h * sizeof(int));
    int* buffer = (int*) malloc(w * h * sizeof(int));

    int* d_Cells;
    int* d_Buffer;
    
    checkCudaErrors(cudaMalloc(&d_Cells, w * h * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Buffer, w * h * sizeof(int)));
    

    char c;
    // now read the grid
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            scanf("%c", &c);
            if (c == '#') 
                cells[i * w + j] = 1;
            else
                cells[i * w + j] = 0;
        }
        scanf("%c", &c); // newlines
    }


    // copy data to d_Cells and d_Buffer;
    checkCudaErrors(cudaMemcpy(d_Cells, cells, w * h * sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_Buffer, d_Cells, w * h * sizeof(int), cudaMemcpyDeviceToDevice));

    int i = 0;
    float angle = 0;
    // Used to get a consistent frame rate
    for (int step=0; step < n; ++step){
        int step_one_indexed = step + 1;
        // update the new array
        dim3 block_size(32, 32);
        dim3 num_blocks((w + block_size.x - 1) / block_size.x, (h + block_size.y - 1) / block_size.y);
        game_of_life<<<num_blocks, block_size>>>(d_Cells, d_Buffer, w, h, A, B, C, m);

        // Swap the vectors so that the updated one is drawn in the next frame.
        int * temp = d_Buffer;
        d_Buffer = d_Cells;
        d_Cells = temp;

        if (step == 0 || (step + 1) % m == 0 || step == n - 1) {
            // copy d_Cells to cells:
            checkCudaErrors(cudaMemcpy(cells, d_Cells, w * h * sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaDeviceSynchronize());
            print(w, h, cells);
        }
    }
    free(cells);
    free(buffer);

    cudaFree(d_Buffer);
    cudaFree(d_Cells);
}