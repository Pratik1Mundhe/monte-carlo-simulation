#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

// Texture memory declaration
texture<int, cudaTextureType1D, cudaReadModeElementType> tex_r;
texture<int, cudaTextureType1D, cudaReadModeElementType> tex_b;

__global__ void calc_points(int *res_r, int *res_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use fast math for optimization
    curandState state;
    curand_init(clock64() + idx, 0, 0, &state);

    int local_r = 0, local_b = 0;

    for (int i = 0; i < (1 << 20); i++) {  // 1e6 → 2^20 (bitwise optimization)
        float x = __fmul_rn(curand_uniform(&state), 1.0f);  
        float y = __fmul_rn(curand_uniform(&state), 1.0f);

        // Avoiding sqrt() using squared distance check
        if (__fadd_rn(__fmul_rn(x, x), __fmul_rn(y, y)) <= 1.0f)
            local_r++;
        else
            local_b++;
    }

    // Use atomicAdd to update results (reading from texture memory)
    atomicAdd(res_r, local_r);
    atomicAdd(res_b, local_b);
}

int main() {
    int threadsPerBlock = 1024;
    int numBlocks = 64;  // Reasonable number to avoid excessive launch overhead

    int h_r = 0, h_b = 0;
    int *d_r, *d_b;

    cudaMalloc((void **)&d_r, sizeof(int));
    cudaMalloc((void **)&d_b, sizeof(int));

    cudaMemcpy(d_r, &h_r, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);

    // Bind texture memory to device variables
    cudaBindTexture(0, tex_r, d_r, sizeof(int));
    cudaBindTexture(0, tex_b, d_b, sizeof(int));

    calc_points<<<numBlocks, threadsPerBlock>>>(d_r, d_b);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_r, d_r, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);

    // Unbind texture memory
    cudaUnbindTexture(tex_r);
    cudaUnbindTexture(tex_b);

    cudaFree(d_r);
    cudaFree(d_b);

    int n = h_r + h_b;
    std::cout << "Estimated Pi: " << (4.0 * h_r) / n << std::endl;

    return 0;
}
