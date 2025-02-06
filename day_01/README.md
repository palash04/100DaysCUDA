# CUDA Memory Allocation and Data Transfer

## Overview

This program demonstrates:
- Allocating memory on the **GPU** using `cudaMalloc`
- Copying data from **CPU to GPU** using `cudaMemcpy`
- Copying data back from **GPU to CPU** using `cudaMemcpy`

## How to Run

1. Ensure you have **CUDA installed** on your system.
2. Compile the code using:
   ```sh
   nvcc -o cuda_memory_transfer cuda_memory_transfer.cu

3. Run the compiled program:
   ```sh
   ./cuda_memory_transfer
