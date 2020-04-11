#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

__global__ void get__min_max(const float* const d_logLuminance, float* const d_comparer, float& min_logLum, float& max_logLum, const size_t numRows, const size_t numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	float localMax = d_logLuminance[thread_1D_pos];
	float localMin = d_logLuminance[thread_1D_pos];

	d_comparer[thread_1D_pos] = d_logLuminance[thread_1D_pos];
	d_comparer[thread_1D_pos + 1] = d_logLuminance[thread_1D_pos];

	__syncthreads();

	for (unsigned int s = (numCols * numRows) * 0.5; s > 0; s >>= 1)
	{
		if (thread_1D_pos < s)
		{
			//Max
			if (d_comparer[thread_1D_pos] < d_comparer[thread_1D_pos + s])
			{
				printf("Nivel %d: %f es mayor que %f\n", s, d_comparer[thread_1D_pos + s], d_comparer[thread_1D_pos]);
				d_comparer[thread_1D_pos] = d_comparer[thread_1D_pos + s];
			}
			else
				printf("Nivel %d: %f es mayor que %f\n", s, d_comparer[thread_1D_pos], d_comparer[thread_1D_pos + s]);
		}
		__syncthreads();
	}
	__syncthreads();

	if (thread_1D_pos == 0)
	{
		max_logLum = d_comparer[thread_1D_pos];
		printf("valor final %f : ", max_logLum);
	}

	printf("Valor logLuminance %f \n", d_logLuminance[thread_1D_pos]);
}

void calculate_cdf(const float* const d_logLuminance,
	unsigned int* const d_cdf,
	float& min_logLum,
	float& max_logLum,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{
	size_t numPixels = numRows * numCols;
	const dim3 blockSize(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
	const dim3 gridSize(ceil((float)numCols / BLOCK_WIDTH), ceil((float)numRows / BLOCK_HEIGHT), 1);

	// TODO
	//1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance

	//Crear valor intermedio para trabajar
	float* d_comparer;
	cudaMalloc(&d_comparer, sizeof(float) * numPixels);
	cudaMemset(d_comparer, 0, sizeof(float) * numPixels);
	//std::cout << "antes" << std::endl;
	get__min_max << < gridSize, blockSize >> > (d_logLuminance, d_comparer, min_logLum, max_logLum, numRows, numCols);
	//std::cout << "despues" << std::endl;

	//2) Obtener el rango a representar
	//3) Generar un histograma de todos los valores del canal logLuminance usando la formula
	//bin = (Lum [i] - lumMin) / lumRange * numBins
	//4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf)
	//de los valores de luminancia. Se debe almacenar en el puntero c_cdf

//	checkCudaErrors(cudaFree(d_comparer));
}