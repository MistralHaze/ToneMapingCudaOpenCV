#include <device_functions.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#include <iostream>
#include <iomanip>

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file,
	const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

void findMinMaxLuminancy(const float* const d_logLuminance,
	float& min_logLum,
	float& max_logLum,
	const size_t numRows,
	const size_t numCols,
	const dim3& blockSize,
	const dim3& gridSize);

__global__ void get_min_max(const float* const d_logLuminance, float* d_minLogLumPerBlock, float* const d_maxLogLumPerBlock, const size_t numRows, const size_t numCols)
{
	__shared__ float sharedMax[BLOCK_WIDTH * BLOCK_HEIGHT];
	__shared__ float sharedMin[BLOCK_WIDTH * BLOCK_HEIGHT];

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	// 0 - total threads en ejecucion
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	// 0 - num threads por bloque
	const int shared_pos = threadIdx.y * blockDim.x + threadIdx.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
	{
		sharedMax[shared_pos] = 0.0f;
		sharedMin[shared_pos] = INFINITY;
		return;
	}
	else
	{
		sharedMax[shared_pos] = d_logLuminance[thread_1D_pos];
		sharedMin[shared_pos] = sharedMax[shared_pos];
	}

	__syncthreads();

	for (unsigned int s = (blockDim.x * blockDim.y) >> 1; s > 0; s >>= 1)
	{
		if (shared_pos < s)
		{
			//Max
			if (sharedMax[shared_pos] < sharedMax[shared_pos + s])
			{
				sharedMax[shared_pos] = sharedMax[shared_pos + s];
			}

			//Min
			if (sharedMin[shared_pos] > sharedMin[shared_pos + s])
			{
				sharedMin[shared_pos] = sharedMin[shared_pos + s];
			}
		}
		__syncthreads();
	}

	if (shared_pos == 0)
	{
		const unsigned int blockId = blockIdx.y * ceil((float)numCols / BLOCK_WIDTH) + blockIdx.x;

		d_maxLogLumPerBlock[blockId] = sharedMax[0];
		d_minLogLumPerBlock[blockId] = sharedMin[0];
	}
}

/*
* This function is intended to work with the output of get_min_max.
* Since that function outputs max values of each block in an array, we have to find
* the max value from there. This kernel helps if the array is too big to search it
* efficiently in CPU.
*/
__global__ void get_min_max_aux(float* const d_minLogLumPerBlock, float* const d_maxLogLumPerBlock, const size_t size)
{
	__shared__ float sharedMax[BLOCK_WIDTH * BLOCK_HEIGHT];
	__shared__ float sharedMin[BLOCK_WIDTH * BLOCK_HEIGHT];

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	// 0 - total threads en ejecucion
	const int thread_1D_pos = thread_2D_pos.x * blockDim.y + thread_2D_pos.y;

	// 0 - num threads por bloque
	const int shared_pos = threadIdx.y * blockDim.x + threadIdx.x;

	if (thread_1D_pos >= size)
	{
		sharedMax[shared_pos] = 0.0f;
		sharedMin[shared_pos] = INFINITY;
		return;
	}
	else
	{
		sharedMax[shared_pos] = d_maxLogLumPerBlock[thread_1D_pos];
		sharedMin[shared_pos] = d_minLogLumPerBlock[thread_1D_pos];
	}

	__syncthreads();

	for (unsigned int s = (blockDim.x * blockDim.y) >> 1; s > 0; s >>= 1)
	{
		if (shared_pos < s)
		{
			//Max
			if (sharedMax[shared_pos] < sharedMax[shared_pos + s])
			{
				sharedMax[shared_pos] = sharedMax[shared_pos + s];
			}

			//Min
			if (sharedMin[shared_pos] > sharedMin[shared_pos + s])
			{
				sharedMin[shared_pos] = sharedMin[shared_pos + s];
			}
		}
		__syncthreads();
	}

	if (shared_pos == 0)
	{
		const unsigned int blockId = blockIdx.y * ceil((float)size / BLOCK_WIDTH) + blockIdx.x;

		d_maxLogLumPerBlock[blockId] = sharedMax[0];
		d_minLogLumPerBlock[blockId] = sharedMin[0];
	}
}

__global__ void readHisto(unsigned int* const d_histogram, const size_t numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	// 0 - total threads en ejecucion
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//All threads handle blockDim.x* gridDim.x  consecutive elements
	if (thread_2D_pos.x >= numCols) return;

	if (thread_1D_pos == 0)
	{
		int sumtotal = 0;
		printf("histogram values \n");
		for (int i = 0; i < 1024; i++)
		{
			printf("Bin %d value %d \n", i, d_histogram[i]);
			sumtotal += d_histogram[i];
		}
		printf("sumtotla %d  \n", sumtotal);
	}
}

__global__ void generateHistogram(const float* const d_logLuminance, const size_t numBins, unsigned int* const d_histogram,
	const float luminanceRange, const float min_logLum, const size_t numRows, const size_t numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	// 0 - total threads en ejecucion
	//const int thread_1D_pos = thread_2D_pos.x * numCols + thread_2D_pos.y;
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//All threads handle blockDim.x* gridDim.x  consecutive elements
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

	//bin = (Lum[i] - lumMin) / lumRange * numBins;
	//numBins value is 1024 so we need to settle the range 0 - 1023
	unsigned int bin = roundf((d_logLuminance[thread_1D_pos] - min_logLum) / luminanceRange * (numBins - 1));

	if (bin < 0 || bin > 1023)
		printf("Bin %d \n", bin);

	atomicAdd(&(d_histogram[bin]), 1);
}

__global__ void exclusive_scan(unsigned int* const d_histogram, unsigned int* const d_cdf, const size_t numBins, const size_t numRows, const size_t numCols)
{
	__shared__ int temp_array[BLOCK_WIDTH * BLOCK_HEIGHT * 2];

	// 0 - num threads por bloque
	const int shared_pos = threadIdx.y * blockDim.x + threadIdx.x;
	const int id = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int halfSize = numBins * 0.5f;
	int ai = shared_pos;
	int bi = shared_pos + halfSize;

	//Assign shared memory
	temp_array[ai] = d_histogram[shared_pos];
	if (shared_pos + halfSize < numBins)
	{
		temp_array[bi] = d_histogram[shared_pos + halfSize];
	}
	else
	{
		temp_array[bi] = 0;
	}

	unsigned int offset = 1;
	for (unsigned int s = numBins >> 1; s > 0; s >>= 1)
	{
		__syncthreads();
		if (shared_pos < s)
		{
			ai = offset * (2 * shared_pos + 1) - 1;
			bi = offset * (2 * shared_pos + 2) - 1;
			temp_array[bi] += temp_array[ai];
		}

		offset <<= 1;
	}

	if (shared_pos == 0)
	{
		temp_array[numBins - 1] = 0;
	}

	for (int s = 1; s < numBins; s <<= 1)
	{
		offset >>= 1;

		__syncthreads();
		if (shared_pos < s)
		{
			ai = offset * (2 * shared_pos + 1) - 1;
			bi = offset * (2 * shared_pos + 2) - 1;
			int temp = temp_array[ai];
			temp_array[ai] = temp_array[bi];
			temp_array[bi] += temp;
		}
	}

	__syncthreads();

	d_cdf[shared_pos] = temp_array[shared_pos];
	d_cdf[shared_pos + halfSize] = temp_array[shared_pos + halfSize];

	if (shared_pos == 0)
	{
		printf("Accumulated cdf: \n");

		for (int i = 0; i < 1024; i++)
		{
			printf("i: %d, histvalue: %d, cdf : %d \n", i, d_histogram[i], d_cdf[i]);
			printf("Bin %d value %d \n", i, d_histogram[i]);
		}
	}
}

void calculate_cdf(const float* const d_logLuminance,
	unsigned int* const d_cdf,
	float& min_logLum,
	float& max_logLum,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{
	const dim3 blockSize(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
	const dim3 gridSize(ceil((float)numCols / BLOCK_WIDTH), ceil((float)numRows / BLOCK_HEIGHT), 1);

	// TODO
	//1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance
	//Crear valor intermedio para trabajar
	findMinMaxLuminancy(d_logLuminance, min_logLum, max_logLum, numRows, numCols, blockSize, gridSize);

	std::cout << "max " << max_logLum << " min " << min_logLum << std::endl;

	//2) Obtener el rango a representar
	unsigned int* d_histogram;
	checkCudaErrors(cudaMalloc(&d_histogram, sizeof(unsigned int) * numBins));
	checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(unsigned int) * numBins));

	float luminanceRange = max_logLum - min_logLum;

	if (luminanceRange == 0)
	{
		std::cerr << "Computed range of luminance was 0. Returning" << std::endl;
		return;
	}

	//3) Generar un histograma de todos los valores del canal logLuminance usando la formula
	//bin = (Lum [i] - lumMin) / lumRange * numBins
	generateHistogram << <gridSize, blockSize >> > (d_logLuminance, numBins, d_histogram, luminanceRange, min_logLum, numRows, numCols);

	readHisto << <dim3(1, 1, 1), blockSize >> > (d_histogram, numCols);

	//4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf)
	//de los valores de luminancia. Se debe almacenar en el puntero c_cdf
	exclusive_scan << <dim3(1, 1, 1), blockSize >> > (d_histogram, d_cdf, numBins, numRows, numCols);
}

void findMinMaxLuminancy(const float* const d_logLuminance,
	float& min_logLum,
	float& max_logLum,
	const size_t numRows,
	const size_t numCols,
	const dim3& blockSize,
	const dim3& gridSize)
{
	float* d_maxLogLumPerBlock, * d_minLogLumPerBlock;
	float* d_maxLogLum, * d_minLogLum;

	checkCudaErrors(cudaMalloc(&d_maxLogLumPerBlock, sizeof(float) * gridSize.x * gridSize.y));
	checkCudaErrors(cudaMalloc(&d_minLogLumPerBlock, sizeof(float) * gridSize.x * gridSize.y));

	checkCudaErrors(cudaMalloc(&d_maxLogLum, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_minLogLum, sizeof(float)));

	get_min_max << < gridSize, blockSize >> > (d_logLuminance, d_minLogLumPerBlock, d_maxLogLumPerBlock, numRows, numCols);

	//Obtener max y min a partir de los max min de cada bloque
	int threadsPerBlock = BLOCK_WIDTH * BLOCK_HEIGHT;
	int numThreads = gridSize.x * gridSize.y;
	while (numThreads > 1)
	{
		int numBlocks = ceil((float)numThreads / threadsPerBlock);
		dim3 gridSizeAux(numBlocks, 1, 1);

		get_min_max_aux << < gridSizeAux, blockSize >> > (d_minLogLumPerBlock, d_maxLogLumPerBlock, numThreads);

		numThreads = numBlocks;
	}

	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy((void*)&max_logLum, (void*)&d_maxLogLumPerBlock[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((void*)&min_logLum, (void*)&d_minLogLumPerBlock[0], sizeof(float), cudaMemcpyDeviceToHost));

	std::cout << "valor final tras kernel Max: " << max_logLum << std::endl;
	std::cout << "valor final tras kernel Min: " << min_logLum << std::endl;

	checkCudaErrors(cudaFree(d_maxLogLumPerBlock));
	checkCudaErrors(cudaFree(d_minLogLumPerBlock));
	//checkCudaErrors(cudaFree(d_maxLogLum));
	//checkCudaErrors(cudaFree(d_minLogLum));
}