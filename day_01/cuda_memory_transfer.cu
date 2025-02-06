#include <iostream>
#include <cuda_runtime.h>

int main() {

    int *d_a;  // CPU pointer
    int h_a = 100;
    int h_result; // variable to copy back from GPU

    std::cout << "Address of d_a: " << &d_a << std::endl; // cpu memory address
    std::cout << "Value at d_a (before cudaMalloc): " << d_a << std::endl;   // some random cpu address
    // std::cout << "Value d_a is pointing to: " << *d_a << std::endl;   // segmentation fault - since it is not pointing to anywhere yet.

    // Allocate memory to GPU
    cudaMalloc((void**)&d_a, sizeof(int)); // address of allocated memory is stored in d_a

    // Address of d_a
    std::cout << "Address of d_a: " << &d_a << std::endl; // remains same
    // Value at d_a
    std::cout << "Value at d_a (GPU Address): " << d_a << std::endl;   // has gpu allocated address 

    // Now copy the value from host address to gpu allocated address which is stored in d_a
    // cpu to gpu
    cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
    //std::cout << "Value d_a is pointing to: " << *d_a << std::endl;   // segmaCannot directly dereference gpu address from cpu

    // gpu to cpu
    cudaMemcpy(&h_result, d_a, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Value at h_result (GPU to CPU copied value): " << h_result << std::endl;

    return 0;
}


/*

Output:
-----------------------------------
Address of d_a: 0x7ffd34823270
Value at d_a (before cudaMalloc): 0
Address of d_a: 0x7ffd34823270
Value at d_a (GPU Address): 0x7fca59e00000
GPU to CPU copied value: 100

*/
