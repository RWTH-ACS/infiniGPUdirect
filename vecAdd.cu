#include <cuda.h>

__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N)
    {
        C[i] = A[i] + B[i];
    } 
}

extern "C" int vecAdd(int* d_A, int* d_B, int* d_C, int N)
{
    int threadsPerBlock = 256;  
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    return 0;
}
