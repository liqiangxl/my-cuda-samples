#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void dynamic_shared_memory_test(int* input)
{
    extern __shared__ int shared_array[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    shared_array[tid] = input[i];
    __syncthreads();

    // Perform some operation on shared memory data
    shared_array[tid] *= 2;
    __syncthreads();

    // Write data back to global memory
    input[i] = shared_array[tid];
}

int main()
{
    const int N = 256*16*128*128;
    const int BLOCK_SIZE = 128;
    std::vector<int> input(N,0), output(N,0);
    int* d_input = nullptr;

    int dev_id = 0;
    cudaSetDevice(dev_id);

    // cudaFuncCache cache_config = cudaFuncCachePreferShared;//cudaFuncCachePreferL1;
    cudaFuncCache cache_config = cudaFuncCachePreferL1;//cudaFuncCachePreferL1;
    cudaDeviceSetCacheConfig(cache_config);

    cudaFuncCache current_cache_config;
    cudaDeviceGetCacheConfig(&current_cache_config);
    std::cout << "Current cache configuration: " << current_cache_config << std::endl;

    
    // Allocate device memory
    cudaMalloc((void**)&d_input, sizeof(int) * N);

    // Copy input data to device memory
    cudaMemcpy(d_input, input.data(), sizeof(int) * N, cudaMemcpyHostToDevice);

    // Launch kernel with dynamic shared memory allocation
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(N / BLOCK_SIZE);
    int shared_memory_size = BLOCK_SIZE * sizeof(int);
    dynamic_shared_memory_test<<<dimGrid, dimBlock, shared_memory_size>>>(d_input);

    // Copy output data from device memory
    cudaMemcpy(output.data(), d_input, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Print output
    for (int i = 0; i < N; i++)
    {
    }

    // Free device memory
    cudaFree(d_input);

    return 0;
}
