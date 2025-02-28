#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

// CUDA Kernel to calculate points inside and outside the unit circle
// using Monte Carlo method with atomic addition
__global__ void calc_points(unsigned long long *res_r, unsigned long long *res_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Unique thread index

    // Initialize a random number generator for this thread
    curandState state;
    curand_init(clock64() + idx, 0, 0, &state);

    unsigned long long local_r = 0, local_b = 0; // Local counters for this thread

    // Monte Carlo simulation: throw (1 << 20) random points per thread
    for (int i = 0; i < (1 << 20); i++) { // 2^20 = 1,048,576 points per thread
        float x = __fmul_rn(curand_uniform(&state), 1.0f); // Random x in [0,1]
        float y = __fmul_rn(curand_uniform(&state), 1.0f); // Random y in [0,1]

        // Check if the point (x, y) falls inside the unit quarter-circle
        // Using the equation x^2 + y^2 <= 1 (avoiding sqrt for performance)
        if (__fadd_rn(__fmul_rn(x, x), __fmul_rn(y, y)) <= 1.0f)
            local_r++; // Inside the circle
        else
            local_b++; // Outside the circle
    }

    // Use atomicAdd to safely accumulate results across threads
    atomicAdd(res_r, local_r);
    atomicAdd(res_b, local_b);
}

int main() {
    int threadsPerBlock = 1024; // Maximum threads per block
    int numBlocks = 64; // Number of blocks in execution

    // Host variables to store the final count of inside and outside points
    unsigned long long h_r = 0, h_b = 0;
    unsigned long long *d_r, *d_b;

    // Allocate memory on the GPU and check for errors
    cudaError_t err;
    err = cudaMalloc((void **)&d_r, sizeof(unsigned long long));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_r: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaMalloc((void **)&d_b, sizeof(unsigned long long));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_b: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize device memory to zero to prevent garbage values
    cudaMemset(d_r, 0, sizeof(unsigned long long));
    cudaMemset(d_b, 0, sizeof(unsigned long long));

    // Launch CUDA kernel with the specified grid and block dimensions
    calc_points<<<numBlocks, threadsPerBlock>>>(d_r, d_b);

    // Check for kernel launch errors
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernelErr) << std::endl;
        return -1;
    }

    // Ensure kernel execution has completed before reading results
    cudaDeviceSynchronize();

    // Copy the results back from device memory to host memory
    err = cudaMemcpy(&h_r, d_r, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for h_r: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaMemcpy(&h_b, d_b, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for h_b: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory to avoid memory leaks
    cudaFree(d_r);
    cudaFree(d_b);

    // Debugging information: print raw counts of points inside and outside
    std::cout << "Inside circle: " << h_r << ", Outside circle: " << h_b << std::endl;

    // Compute total number of points simulated
    unsigned long long n = h_r + h_b;
    if (n == 0) {
        std::cerr << "ERROR: No points were generated! Kernel execution failed.\n";
        return -1;
    }

    // Estimate Pi using the Monte Carlo method:
    // Area of quarter-circle = (pi/4), thus pi ≈ 4 * (points inside / total points)
    std::cout << "Estimated Pi: " << (4.0 * h_r) / n << std::endl;

    return 0;
}