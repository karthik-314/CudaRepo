
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<iostream>
#include<vector>
#include<algorithm>
#include<ctime>

#define THREAD_COUNT 320

using namespace std;

__global__ void random_outlist(bool *cugraph, int n, int *curand1, int *curand2, int f) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int degree, count = 3, temp;

	if (id < n) {
		degree = (curand1[id] + curand2[0])%f;
		if ((curand1[id] + curand2[1]) % 2)	degree *= (curand1[id] + curand2[2]) % 2 + 1;
		while (degree--) {
			temp = (curand1[id] + curand2[count++]) % n;
			if(temp != id)	cugraph[id*n + temp] = 1;
		}
	}

}

__global__ void get_lists(bool *cugraph, int *inlist, int *degree, int f, int n) {
	f = 2 * f + 1;
	int id = blockIdx.x * blockDim.x + threadIdx.x, start, count, start2;

	if (id < n) {
		start = id * f;
		start2 = id * n;
		count = 0;
		for (int i = 0; i < f; i++) {
			inlist[start + i] = -1;
		}
		degree[id] = 0;
		for (int i = 0; i < n; i++) {
			if (cugraph[start2 + i]) {
				degree[id]++;	
			}
		}

		for (int i = 0; i < n; i++) {
			if (cugraph[n*i + id]) {
				inlist[start + count++] = i;
			}
		}

	}

}

__global__ void random_surfer(bool *cugraph, int n, int visit_count) {
	int vis;

}

__global__ void single_rank(int *cuinlist, int *cudegree, double *pages, double *pages2, int f, int n, double d) {
	int id = blockDim.x * blockIdx.x + threadIdx.x, count = 0, start = id*f;
	double rank = (1 - d)/n, temp = 0;

	if (id < n) {
		while (cuinlist[start + count] != -1 && count < f) {
			temp += (pages[cuinlist[start + count]] / cudegree[cuinlist[start + count]]);
			count++;
		}
		temp *= d;
		rank += temp;
		pages2[id] = rank;
	}
}

__global__ void initial_ranks(double *pages, int n) {
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	if (id < n)	pages[id] = (double)1 / n;

}

void pagerank(int *cuinlist, int *cudegree, double *cupages, double *cupages2, int n, int f) {
	double d = 0.85;
	int iter = 13;


	for (int i = 0; i < iter; i++) {
		if (i % 2) {
			single_rank<<<n/THREAD_COUNT + 1, THREAD_COUNT>>>(cuinlist, cudegree, cupages, cupages2, 2*f + 1, n, d);
		}
		else {
			single_rank << <n / THREAD_COUNT + 1, THREAD_COUNT >> >(cuinlist, cudegree, cupages2, cupages, 2*f + 1, n, d);
		}
	}
}

double* naive_pagerank(bool *graph, int n) {
	double *ranks = (double*)malloc(n * sizeof(double)); //Stores all the page ranks
	double d = 0.85; //Damping Factor = 0.85
	double temp;
	vector<int> outbounds = *new vector<int>(n, 0); //Number of outbound links for each vertex
													//List of inbound links for each vertex
	vector<vector<int>> inbounds = *new vector<vector<int>>(n, *new vector<int>());
	//Initializing the outbound, inbound and rank list.
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (graph[n*i + j]) {
				outbounds[i]++;
				inbounds[j].push_back(i);
			}
		}
		ranks[i] = 1 / n;
	}
	//Calculating pagerank. Number of iterations = 7
	for (int i = 0; i < 13; i++) {
		for (int j = 0; j < n; j++) {
			temp = 0;
			for (int k = 0; k < inbounds[j].size(); k++)
				temp += ranks[inbounds[j][k]] / outbounds[inbounds[j][k]];
			ranks[j] = (1 - d) / n + d * temp;
		}
	}
	return ranks;
}

float gpu_pagerank(bool *cugraph, int n, int f) {
	double *cupages, *cupages2, *pages;
	int *cuinlist, *cudegree;
	float gpu = 0;
	cudaEvent_t start, stop;

	cudaMallocManaged(&cuinlist, n*(2 * f + 1) * sizeof(int));
	cudaMallocManaged(&cudegree, n * sizeof(int));

	cudaMallocManaged(&cupages, n * sizeof(double));
	cudaMallocManaged(&cupages2, n * sizeof(double));
	pages = (double*)malloc(n * sizeof(double));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	get_lists << <n / THREAD_COUNT + 1, THREAD_COUNT >> > (cugraph, cuinlist, cudegree, f, n);
	initial_ranks << <n / THREAD_COUNT + 1, THREAD_COUNT >> > (cupages, n);
	pagerank(cuinlist, cudegree, cupages, cupages2, n, f);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(pages, cupages2, n * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(cuinlist);
	cudaFree(cudegree);
	cudaFree(cupages);
	cudaFree(cupages2);
	free(pages);

	return gpu;
}

double cpu_pagerank(bool *graph, int n) {
	clock_t begin, end;
	double *ranks = (double*)malloc(n * sizeof(double)), cpu;
	begin = clock();

	ranks = naive_pagerank(graph, n);

	end = clock();

	cpu = (double)(end - begin) / CLOCKS_PER_SEC;
	cpu *= 1000;
	free(ranks);
	return cpu;
}

int main() {
	int n = 4000, f = 80, count;
	bool *cugraph, *graph;
	int *rand1, *rand2, *curand1, *curand2, *degree;
	double *ranks, cpu;
	float gpu;
	rand1 = (int*)malloc(n * sizeof(int));
	rand2 = (int*)malloc(n * sizeof(int));

	cudaMallocManaged(&curand1, n * sizeof(int));
	cudaMallocManaged(&curand2, n * sizeof(int));

	for (int i = 0; i < n; i++) {
		rand1[i] = i;
		rand2[i] = i;
	}

	random_shuffle(rand1, rand1 + n - 1);
	random_shuffle(rand2, rand2 + n - 1);

	cudaMemcpy(curand1, rand1, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(curand2, rand2, n * sizeof(int), cudaMemcpyHostToDevice);
	//Need to check this logic

	cudaMallocManaged(&cugraph, n * n * sizeof(bool));
	graph = (bool*)malloc(n * n * sizeof(bool));

	random_outlist<<<n/THREAD_COUNT + 1, THREAD_COUNT>>>(cugraph, n, curand1, curand2, f);

	cudaFree(curand1);
	cudaFree(curand2);
	free(rand1);
	free(rand2);

	cudaMemcpy(graph, cugraph, n * n * sizeof(bool), cudaMemcpyDeviceToHost);

	gpu = 0;
	for(int i = 0 ; i < 5 ; i++)	gpu += gpu_pagerank(cugraph, n, f);
	gpu /= 5;

	cudaFree(cugraph);

	//for (int i = 0; i < n; i++)	cout << i << "\t" << pages[i] << "\n";

	cpu = 0;
	for(int i = 0 ; i < 5 ; i++)	cpu += cpu_pagerank(graph, n);
	cpu /= 5;

	cout << cpu << "ms\n";
	cout << gpu << "ms\n";
	cout << "Speed Up:    " << (cpu / ((double)gpu)) << "\n";

	//for (int i = 0; i < n; i++)	cout << i << "\t" << pages[i] << "\t" << ranks[i] << "\t" << "\n";

	return 0;
}