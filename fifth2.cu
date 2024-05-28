#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#define N 214748364
#define M 8 //Watki na blok

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

void add_cpu(int *a, int *b, int *c, int n)
{
	for(int i = 0; i < n; i++)
	{
		c[i] = a[i] + b[i];
	}
}

int main(void) {
	int *a, *b, *c; // host copies of a, b, c
	int *d_a, *d_b, *d_c; // device copies of a, b, c
	int size = N * sizeof(int);
	//int i;
	srand(time(NULL));

	a = (int *)malloc(size); random(a, N);
	b = (int *)malloc(size); random(b, N);
	c = (int *)malloc(size);

	// measuring time for cpu
	clock_t start_cpu = clock();
    	add_cpu(a, b, c, N);
    	clock_t end_cpu = clock();
    	double cpu_time = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;

	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	// Alloc space for host copies of a, b, c and setup input values
	random(a, N);
	random(b, N);
	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Begin timer without copying
	
	cudaEvent_t start_gpu_comp, stop_gpu_comp;
    	cudaEventCreate(&start_gpu_comp);
    	cudaEventCreate(&stop_gpu_comp);
    	cudaEventRecord(start_gpu_comp, 0);

	// Launch add() kernel on GPU
	add<<<(N+M-1)/M,M>>>(d_a, d_b, d_c, N);

	// timer 
	cudaEventRecord(stop_gpu_comp, 0);
    	cudaEventSynchronize(stop_gpu_comp); 
    	float gpu_comp_time;
    	cudaEventElapsedTime(&gpu_comp_time, start_gpu_comp, stop_gpu_comp);

	// Begin timer with copying

	cudaEventCreate(&start_gpu_comp);
    	cudaEventCreate(&stop_gpu_comp);
    	cudaEventRecord(start_gpu_comp, 0);
	
	// Launch add() kernel on GPU
	add<<<(N+M-1)/M,M>>>(d_a, d_b, d_c, N);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	
	// timer 
	cudaEventRecord(stop_gpu_comp, 0);
    	cudaEventSynchronize(stop_gpu_comp); 
    	float gpu_total_time;
    	cudaEventElapsedTime(&gpu_total_time, start_gpu_comp, stop_gpu_comp);

	printf("TOTAL SIZE : %d\n", N);
	printf("CPU time: %f milliseconds\n", cpu_time * 1000);
    	printf("GPU computation time (without memory copy): %f milliseconds\n", gpu_comp_time);
    	printf("Total GPU time (with memory copy): %f milliseconds\n", gpu_total_time);
	
	// Cleanup
	//printf("%d+%d=%d\n",a,b,c);
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}


