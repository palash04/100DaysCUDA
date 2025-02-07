/*

Vector addtion

1. Allocate memory for vectors on host and device.
2. Copy the input vectors from host to device.
3. Launch the CUDA kernel to compute the vector addition in parallel.
4. Copy the result from the device back to the host.
5. Free the allocated memory

*/

#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    // __global__ means function runs on gpu
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {

    int k = 20;
    int n = 1 << k;    // 2^(k) elements

    size_t size = sizeof(float) * n;  // (4 * n) bytes

    // Allocate memory on the host
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // initialize input vectors
    for (int i = 0; i < n; i++) {
        h_A[i] = static_cast<float>(i);
        h_A[i] = static_cast<float>(i * 2);
    }
    
    // Allocate memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // define grid and block sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // verify result
    bool success = true;
    for (int i = 0; i < n ; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            success = false;
            std::cout << "Error at index: " << i << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            break;
        }
    }
    
    if (success) {
        std::cout << "Vector addition successful" << std::endl;
    }

    // free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
