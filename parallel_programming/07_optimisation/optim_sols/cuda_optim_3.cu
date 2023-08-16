#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include "./common/helper_cuda.h"
#include <cuda_runtime.h>
#define IDX(y, x) ((y) * W + (x))
#define BLOCK_SIZE 16

__device__ int neighbours_shared(int x, int y, int W, int H, bool* cells) {
    int N = 0;
    int tx, ty;
    // for all neighbours
    #pragma unroll
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            // not the cell itself
            if (!i && !j) continue;
            tx = x + i;
            ty = y + j;

            // add if the cell is active
            int idx = ty * W + tx;
            N += cells[idx];
        }
    }
    return N;
}


void print(int W, int H, bool* cells) {
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

__global__ void game_of_life(bool* d_Cells, bool* d_Buffer, int w, int h, int A, int B, int C, int internal_iters) {
    __shared__ bool data_of_the_things[(BLOCK_SIZE + 2) * (BLOCK_SIZE + 2)]; // copy data now

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int testx = min(BLOCK_SIZE, w);
    int testy = min(BLOCK_SIZE, h);

    int TEST = BLOCK_SIZE + 2;

    if (x < w && y < h){
        int my_tx = threadIdx.x + 1;
        int my_ty = threadIdx.y + 1;
        int idx = y * w + x;

        // copy main data

        data_of_the_things[my_ty * TEST + my_tx] = d_Cells[idx];


        if (threadIdx.x == 0){
            data_of_the_things[my_ty * TEST + 0] = d_Cells[y * w + (x - 1 + w) % w];
        }
        if (threadIdx.y == 0){
            data_of_the_things[0 * TEST + my_tx] = d_Cells[((y - 1 + h) % h) * w + x];
        }
        if (threadIdx.x == testx - 1){
            data_of_the_things[my_ty * TEST + testx + 1] = d_Cells[y * w + (x + 1) % w];
        }
        if (threadIdx.y == testy - 1){
            data_of_the_things[(testy+1) * TEST + my_tx] = d_Cells[((y + 1) % h) * w + x];
        }

        // now do 4 corners
        if (threadIdx.x == 0 && threadIdx.y == 0){
            data_of_the_things[0 * TEST + 0] = d_Cells[((y - 1 + h) % h) * w + (x - 1 + w) % w];
        }
        if (threadIdx.x == testx - 1 && threadIdx.y == 0){
            data_of_the_things[0 * TEST + testx + 1] = d_Cells[((y - 1 + h) % h) * w + (x + 1) % w];
        }
        if (threadIdx.x == 0 && threadIdx.y == testy - 1){
            data_of_the_things[(testy+1) * TEST + 0] = d_Cells[((y + 1) % h) * w + (x - 1 + w) % w];
        }
        if (threadIdx.x == testx - 1 && threadIdx.y == testy - 1){
            data_of_the_things[(testy+1) * TEST + testx + 1] = d_Cells[((y + 1) % h) * w + (x + 1) % w];
        }


        __syncthreads(); // synchronise here.

        int N = neighbours_shared(my_tx, my_ty, TEST, TEST, data_of_the_things);
        bool is_on = d_Cells[idx];
        bool new_val = (is_on && (N >= A && N <= B)) || (!is_on && N == C);

        d_Buffer[idx] = new_val;
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOFBF, 16384 * 16); // https://stackoverflow.com/a/65020999

    int w, h, n, m, A, B, C;
    scanf("%d %d %d %d %d %d %d\n", &w, &h, &n, &m, &A, &B, &C);
    bool* cells  = (bool*) malloc(w * h * sizeof(bool));
    bool* buffer = (bool*) malloc(w * h * sizeof(bool));

    bool* d_Cells;
    bool* d_Buffer;
    
    checkCudaErrors(cudaMalloc(&d_Cells, w * h * sizeof(bool)));
    checkCudaErrors(cudaMalloc(&d_Buffer, w * h * sizeof(bool)));
    

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
    checkCudaErrors(cudaMemcpy(d_Cells, cells, w * h * sizeof(bool), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_Buffer, d_Cells, w * h * sizeof(bool), cudaMemcpyDeviceToDevice));

    int i = 0;
    float angle = 0;
    // Used to get a consistent frame rate
    for (int step=0; step < n; ++step){
        int step_one_indexed = step + 1;
        // update the new array
        dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

        dim3 num_blocks((w + block_size.x - 1) / block_size.x, (h + block_size.y - 1) / block_size.y);
        game_of_life<<<num_blocks, block_size>>>(d_Cells, d_Buffer, w, h, A, B, C, m);
        // Swap the vectors so that the updated one is drawn in the next frame.
        bool* temp = d_Buffer;
        d_Buffer = d_Cells;
        d_Cells = temp;

        if (step == 0 || (step + 1) % m == 0 || step == n - 1) {
            // copy d_Cells to cells:
            checkCudaErrors(cudaMemcpy(cells, d_Cells, w * h * sizeof(bool), cudaMemcpyDeviceToHost));
            print(w, h, cells);
        }
    }
    free(cells);
    free(buffer);

    cudaFree(d_Buffer);
    cudaFree(d_Cells);
}