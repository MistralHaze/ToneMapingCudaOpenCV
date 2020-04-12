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

//////////////////
//Sabemos que el valor maximo es : 2.406540 
/////////////////

__global__ void get__min_max(const float* const d_logLuminance, float min_logLum, float* const d_maxLogLumPerBlock, const size_t numRows, const size_t numCols)
{
	__shared__ float shared[BLOCK_WIDTH * BLOCK_HEIGHT];

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	// 0 - total threads en ejecucion
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	// 0 - num threads por bloque
	const int shared_pos = threadIdx.y * blockDim.x + threadIdx.x;	

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
	{
		shared[shared_pos] = 0.0f;
		return;
	}
	else
		shared[shared_pos] = d_logLuminance[thread_1D_pos];


	__syncthreads();

	for (unsigned int s = (blockDim.x * blockDim.y) * 0.5; s > 0; s >>= 1)
	{

		if (thread_1D_pos < s)
		{
			//Max
			if (shared[shared_pos] < shared[shared_pos + s])
			{
				//printf("Id1: %d, Id2: %d, Nivel %d: %f es mayor que %f\n", shared_pos + s, shared_pos, s, shared[shared_pos + s], shared[shared_pos]);
				shared[shared_pos] = shared[shared_pos + s];
			}
			//else
				//printf("Id1: %d, Id2: %d, Nivel %d: %f es mayor que %f\n", shared_pos, shared_pos + s, s, shared[shared_pos], shared[shared_pos + s]);
		}

		__syncthreads();

	}

	__syncthreads();

	if (shared_pos == 0)
	{
		const unsigned int blockId = blockIdx.y * ceil((float)numCols/BLOCK_WIDTH) + blockIdx.x;
		d_maxLogLumPerBlock[blockId] = shared[shared_pos];
		//printf("Thread: %d Block: %d Valor final %f : \n",blockId, thread_1D_pos, d_maxLogLumPerBlock[blockId]);
	}

}
/*
* This function is intended to work with the output of get_min_max. Since that function outputs max values of blocks in an array we have to find
* the max value from there. If the array is small that value can be searched in the CPU. This kernel helps if the array is too big to search it
* efficiently in CPU. 
*/
__global__ void get_min_max_aux(float* const d_maxLogLumPerBlock, const size_t size)
{
	__shared__ float shared[BLOCK_WIDTH * BLOCK_HEIGHT];

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	// 0 - num threads por bloque
	const int shared_pos = threadIdx.y * blockDim.x + threadIdx.x;

	if (shared_pos >= size)
	{
		shared[shared_pos] = 0.0f;
		return;
	}
	else
		shared[shared_pos] = d_maxLogLumPerBlock[shared_pos];

	__syncthreads();

	for (unsigned int s = (blockDim.x * blockDim.y) * 0.5; s > 0; s >>= 1)
	{

		if (shared_pos < s)
		{
			//Max
			if (shared[shared_pos] < shared[shared_pos + s])
			{
				//printf("Id1: %d, Id2: %d, Nivel %d: %f es mayor que %f\n", shared_pos + s, shared_pos, s, shared[shared_pos + s], shared[shared_pos]);
				shared[shared_pos] = shared[shared_pos + s];
			}
			//else
				//printf("Id1: %d, Id2: %d, Nivel %d: %f es mayor que %f\n", shared_pos, shared_pos + s, s, shared[shared_pos], shared[shared_pos + s]);
		}

		__syncthreads();

	}

	if (shared_pos == 0)
	{
		const unsigned int blockId = blockIdx.y * ceil((float)size / BLOCK_WIDTH) + blockIdx.x;
		d_maxLogLumPerBlock[blockId] = shared[shared_pos];
		printf("Thread: %d Block: %d Valor final %f : \n",blockId, shared_pos, d_maxLogLumPerBlock[blockId]);
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
	size_t numPixels = numRows * numCols;
	const dim3 blockSize(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
	const dim3 gridSize(ceil((float)numCols / BLOCK_WIDTH), ceil((float)numRows / BLOCK_HEIGHT), 1);

	// TODO
	//1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance
	//Crear valor intermedio para trabajar
	float* d_maxLogLumPerBlock;
	float* d_maxLogLum;
	checkCudaErrors(cudaMalloc(&d_maxLogLumPerBlock, sizeof(float) * gridSize.x * gridSize.y));
	checkCudaErrors(cudaMalloc(&d_maxLogLum, sizeof(float)));

	checkCudaErrors(cudaMemcpy((void*)d_maxLogLum, (void*)&max_logLum, sizeof(float), cudaMemcpyHostToDevice));

	//checkCudaErrors(cudaMemset(d_comparer, 0, sizeof(float) * numPixels));

	get__min_max <<< gridSize, blockSize >>> (d_logLuminance, min_logLum, d_maxLogLumPerBlock, numRows, numCols);
	get_min_max_aux << < dim3(1, 1, 1), blockSize >> > (d_maxLogLumPerBlock, gridSize.x * gridSize.y);
	//checkCudaErrors(cudaMemcpy((void*)&max_logLum, (void*)d_maxLogLum, sizeof(float), cudaMemcpyDeviceToHost));

	std::cout << "valor final tras kernel " << max_logLum << std::endl;

	//2) Obtener el rango a representar
	//3) Generar un histograma de todos los valores del canal logLuminance usando la formula
	//bin = (Lum [i] - lumMin) / lumRange * numBins
	//4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf)
	//de los valores de luminancia. Se debe almacenar en el puntero c_cdf

//	checkCudaErrors(cudaFree(d_comparer));
}


/*
__global__ void get__min_max_Legacy(const float* const d_logLuminance, float min_logLum, float* d_max_logLum, const size_t numRows, const size_t numCols)
{
	__shared__ float shared[BLOCK_WIDTH][BLOCK_HEIGHT];

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	// 0 - total threads en ejecucion
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	// 0 - num threads por bloque
	//const int shared_pos = threadIdx.y * blockDim.x + threadIdx.x;

	//printf("id thread %d, id block %d - %d \n", shared_pos, blockIdx.x, blockIdx.y);

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
	{
		shared[threadIdx] = 0.0f;
		printf("Id: %d, val: %f \n", shared_pos, shared[shared_pos]);
		return;
	}

	shared[shared_pos] = d_logLuminance[thread_1D_pos];

	printf("Id: %d, val: %f \n", shared_pos, shared[shared_pos]);

	__syncthreads();

	float max = 0.0f;
	for(unsigned int i = 0; i < (blockDim.x * blockDim.y); ++i)
	{
		if (max < shared[i])
			max = shared[i];
	}
	
	if(threadIdx.x == 0)
	{
		printf(" bloque %d, el valor maximo es: %f \n", blockIdx.y * (numCols/BLOCK_WIDTH) + blockIdx.x,  max );
	}
	
	
	for (unsigned int s = (blockDim.x * blockDim.y) * 0.5; s > 0; s >>= 1)
	{

		if (thread_1D_pos < s)
		{
			//Max
			if (shared[thread_1D_pos] < shared[thread_1D_pos + s])
			{
				printf("Id1: %d, Id2: %d, Nivel %d: %f es mayor que %f\n", thread_1D_pos + s, thread_1D_pos, s, shared[thread_1D_pos + s], shared[thread_1D_pos]);
				shared[thread_1D_pos] = shared[thread_1D_pos + s];
			}
			else
				printf("Id1: %d, Id2: %d, Nivel %d: %f es mayor que %f\n", thread_1D_pos, thread_1D_pos + s, s, shared[thread_1D_pos], shared[thread_1D_pos + s]);
		}

		__syncthreads();

		if (thread_1D_pos == 0)
			printf("\n");

	}

	//__syncthreads();

	if (thread_1D_pos == 0)
	{
		*d_max_logLum = shared[thread_1D_pos];
		printf("valor final %f : \n", *d_max_logLum);
	}
	

	//printf("Valor logLuminance %f \n", d_logLuminance[thread_1D_pos]);
}
*/