// The code skeleton was obtained from the wits HPC course, and I then implemented the necessary CUDA / OpenMP functions
// Also see here: https://docs.nvidia.com/cuda/cuda-samples/index.html and here: https://developer.nvidia.com/cuda-example
#include "./common/book.h"
#include "./common/cpu_bitmap.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include <assert.h>
#include "./common/helper_cuda.h"
#include <cuda_runtime.h>
#define DIM 2048
#define COMPLEX_TYPE float
// Complex Class
struct cuComplex
{
    COMPLEX_TYPE r;
    COMPLEX_TYPE i;
    cuComplex(COMPLEX_TYPE a, COMPLEX_TYPE b) : r(a), i(b) {}
    COMPLEX_TYPE magnitude2(void) { return r * r + i * i; }
    cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

// Julia code on cpu
float julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    int MAX_ITER = 100;
    for (i = 0; i < MAX_ITER; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 4)
        {
            i++;
            break;
        }
    }
    if (i == MAX_ITER)
    {
        return 0;
    }
    else
    {
        return (float)i / (float)MAX_ITER;
    }
    return 1;
}

// GPU class
struct cuComplexGPU
{
    COMPLEX_TYPE r;
    COMPLEX_TYPE i;
    __device__ cuComplexGPU(COMPLEX_TYPE a, COMPLEX_TYPE b) : r(a), i(b) {}
    __device__ COMPLEX_TYPE magnitude2(void)
    {
        return r * r + i * i;
    }
    __device__ cuComplexGPU operator*(const cuComplexGPU &a)
    {
        return cuComplexGPU(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplexGPU operator+(const cuComplexGPU &a)
    {
        return cuComplexGPU(r + a.r, i + a.i);
    }
};

// GPU code
__device__ float julia_device(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplexGPU c(-0.8, 0.156);
    cuComplexGPU a(jx, jy);

    int i = 0;
    int MAX_ITER = 100;
    for (i = 0; i < MAX_ITER; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 4)
        {
            i++;
            break;
        }
    }
    if (i == MAX_ITER)
    {
        return 0;
    }
    else
    {
        return (float)i / (float)MAX_ITER;
    }
    return 1;
}

// Save the results to an image
void save(CPUBitmap &bmp, std::string name="out")
{
    float t = omp_get_wtime();
    std::ofstream ofs; // save the framebuffer to file
    ofs.open("./" + name + ".ppm");
    ofs << "P6\n"
        << bmp.x << " " << bmp.y << "\n255\n";
    for (size_t i = 0; i < bmp.x * bmp.y; ++i)
    {
        for (size_t j = 0; j < 3; j++)
        {
            ofs << (char)bmp.pixels[i * 4 + j];
        }
    }
    ofs.close();
}

void kernel(unsigned char *ptr)
{
    // serial
    for (int y = 0; y < DIM; y++)
    {
        for (int x = 0; x < DIM; x++)
        {
            int offset = x + y * DIM;

            float juliaValue = julia(x, y);
            int val = (int)(255 * juliaValue);
            ptr[offset * 4 + 0] = val;
            ptr[offset * 4 + 1] = val;
            ptr[offset * 4 + 2] = 0;
            ptr[offset * 4 + 3] = 0;
        }
    }
}

void kernel_omp(unsigned char *ptr)
{
#pragma omp parallel for shared(ptr)
    for (int y = 0; y < DIM; y++)
    {
        for (int x = 0; x < DIM; x++)
        {
            int offset = x + y * DIM;

            float juliaValue = julia(x, y);
            int val = (int)(255 * juliaValue);
            ptr[offset * 4 + 0] = val;
            ptr[offset * 4 + 1] = val;
            ptr[offset * 4 + 2] = 0;
            ptr[offset * 4 + 3] = 0;
        }
    }
}

__global__ void cuda_julia(unsigned char *a)
{
    // compute the global element index this thread should process
    int x = blockIdx.x;
    int y = blockIdx.y;

    float jul = julia_device(x, y);
    int val = (int)(255 * jul);
    int i = x + y * DIM;
    a[i * 4 + 0] = val;
    a[i * 4 + 1] = val;
    a[i * 4 + 2] = 0;
    a[i * 4 + 3] = 0;
}

// sets up the CUDA memory and runs it in parallel
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


int main(void)
{
    CPUBitmap bitmap_serial(DIM, DIM);
    CPUBitmap bitmap_omp(DIM, DIM);
    CPUBitmap bitmap_cuda(DIM, DIM);

    double startseq = omp_get_wtime();
    kernel(bitmap_serial.get_ptr());
    double endseq = omp_get_wtime();
    double timeSeq = endseq - startseq;

    double startomp = omp_get_wtime();
    kernel_omp(bitmap_omp.get_ptr());
    double endomp = omp_get_wtime();
    double timeomp = endomp - startomp;

    double startcuda = omp_get_wtime();
    unsigned char *dataOnDevice = do_cuda();
    checkCudaErrors(cudaMemcpy(bitmap_cuda.get_ptr(), dataOnDevice, bitmap_cuda.image_size(), cudaMemcpyDeviceToHost));
    double endcuda = omp_get_wtime();
    double timecuda = endcuda - startcuda;

    // make sure is same

    printf("Serial Time: %lfs\n", timeSeq);
    printf("OMP    Time: %lfs\n", timeomp);
    printf("Cuda   Time: %lfs\n", timecuda);

    save(bitmap_cuda, "cuda");
    cudaFree(dataOnDevice);
}
