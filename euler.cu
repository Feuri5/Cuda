#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000000000
#define BLOCK_SIZE 8
float eulersequential(int n)
{
	float g = 0;
	for(int k = 1; k < n; k++)
	{
		g += 1.0/k;
	}

	g -= log(n);

	return g;
}


__global__ void eulerp(float* f)
{
	__shared__ float div[N];
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	div[index] = 1.0/index;
	 __syncthreads();
	 float gamma = 0;
	 for(int i = 0; i < N; i++)
	 {
		 gamma+=div[i];
	 }
	 r = gamma - log(N);
	 *f = r;


}


int main()
{
	clock_t t1, t2;
	t1 = clock();
	float g = eulersequential(N);
	t2 = clock();
	double global_time = ((double)(t2 - t1)) / CLOCKS_PER_SEC; 
	printf("Approximation of Resulted Gamma = %lf\n", g);
	printf("Time for this (in ms) = %lf\n", global_time * 1000);

	float g2;
	float d_g2;

	eulerp<<<(N + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(&d_g2);
	cudaMemcpy(g2, d_g2, sizeof(float), cudaMemcpyDeviceToHost);
	printf("%lf\n", g2);
}
