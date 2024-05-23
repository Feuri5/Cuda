#include<stdio.h>
#define N 32
#define M 8 //Watki na blok

__global__ void add(int *a, int *b, int *c) 
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

void random (int *tab, int wym )
{	
	int i;
	for(i=0;i<wym;i++)
		tab[i]=rand()%101;
}


int main(void) {
	int *a, *b, *c; // host copies of a, b, c
	int *d_a, *d_b, *d_c; // device copies of a, b, c
	int size = N * sizeof(int);
	int i;
	srand(time(NULL));
	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random(a, N);
	b = (int *)malloc(size); random(b, N);
	c = (int *)malloc(size);
	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	// Launch add() kernel on GPU
	add<<<N/M,M>>>(d_a, d_b, d_c);
	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for(i=0;i<N;i++)
	{
		printf("a[%d](%d) + b[%d](%d) = c[%d](%d)\n",i,a[i],i,b[i],i,c[i]);
	}
	// Cleanup
	//printf("%d+%d=%d\n",a,b,c);
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}


