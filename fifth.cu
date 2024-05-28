#include<stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#define N 100000000
#define M 8

__global__ void add(int *a, int *b, int *c, int n) 
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index<n)
		c[index] = a[index] + b[index];
}

void random (int *tab, int wym )
{	
	int i;
	for(i=0;i<wym;i++)
		tab[i]=rand()%101;
}


int main(void) {

	clock_t cpu_time1, cpu_time2;

	int *a, *b, *c; // host copies of a, b, c
	int *d_a, *d_b, *d_c; // device copies of a, b, c
	
	int size = N * sizeof(int);
	srand(time(NULL));
	
	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	
	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random(a, N);
	b = (int *)malloc(size); random(b, N);
	c = (int *)malloc(size);
	
	cpu_time1 = clock();
	for(int i = 0; i < N; i++)
	{
		c[i] = a[i] + b[i];
	
	}
	cpu_time2 = clock();
	double cpu_time = ((double) (cpu_time2 - cpu_time1)) / CLOCKS_PER_SEC;
	

	cudaEvent_t c1, c2;
	cudaEventCreate(&c1);
	cudaEventCreate(&c2);
	
	cudaEvent_t c3, c4;
	cudaEventCreate(&c3);
	cudaEventCreate(&c4);

	cudaEventRecord(c3);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	cudaEventRecord(c1);
	add<<<(N+M-1)/M,M>>>(d_a, d_b, d_c, N);
	cudaEventRecord(c2);
	cudaEventSynchronize(c2);

	float gpu_time_without_copy = 0;

	cudaEventElapsedTime(&gpu_time_without_copy, c1, c2);


	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	cudaEventRecord(c4);
	cudaEventSynchronize(c4);
	float gpu_time = 0;
	cudaEventElapsedTime(&gpu_time, c3, c4);
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);


	printf("Size = %d\n", N);
	printf("Time In CPU in milliseconds = %lf\n", cpu_time * 1000);
	printf("Time In GPU without copying in milliseconds = %lf\n", gpu_time_without_copy);
	printf("Time In GPU with copying in milliseconds = %lf\n", gpu_time);

	return 0;
}


