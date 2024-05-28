#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

#define N 1000000 // Reduced for practical execution
#define BLOCK_SIZE 256

__global__ void eulerp(float* f, int n)
{
    __shared__ float partial_sum[BLOCK_SIZE];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0) partial_sum[threadIdx.x] = 0.0f; // Avoid division by zero
    else partial_sum[threadIdx.x] = (index < n) ? 1.0f / index : 0.0f;
    __syncthreads();

    // Parallel reduction within a block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(f, partial_sum[0]);
}

float eulersequential(int n)
{
    float g = 0;
    for (int k = 1; k < n; k++)
    {
        g += 1.0 / k;
    }
    g -= log(n);
    return g;
}

int main()
{
    // Sequential computation
    clock_t t1, t2;
    t1 = clock();
    float g = eulersequential(N);
    t2 = clock();
    double global_time = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Approximation of Resulted Gamma (CPU) = %lf\n", g);
    printf("Time for this (in ms) = %lf\n", global_time * 1000);

    // GPU computation
    float g2 = 0.0f;
    float *d_g2;
    cudaMalloc((void**)&d_g2, sizeof(float));
    cudaMemcpy(d_g2, &g2, sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    eulerp<<<numBlocks, BLOCK_SIZE>>>(d_g2, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(&g2, d_g2, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_g2);

    g2 -= log(N);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Approximation of Resulted Gamma (GPU) = %lf\n", g2);
    printf("Time for this (in ms) = %lf\n", milliseconds);

    return 0;
}
