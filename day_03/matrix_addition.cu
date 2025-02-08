#include <iostream>
#include <cuda_runtime.h>

#define N 16     // N x N matrix

// CUDA kernel for matrix addition
__global__ void matrixAdd(int *A, int *B, int *C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width and col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }

}

int main() {

    int size = sizeof(int) * N * N;
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;


    // Allocate memory on the host
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize the matrices
    for (int i=0; i<N*N; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid size
    dim3 threadsPerBlock(8,8);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify output
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // free the memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
