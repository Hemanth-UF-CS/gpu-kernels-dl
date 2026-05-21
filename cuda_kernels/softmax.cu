/*
 * Row-wise softmax in CUDA C++.
 *
 * Implements a numerically-stable softmax:
 *   out[i][j] = exp(x[i][j] - max_i) / sum_k exp(x[i][k] - max_i)
 *
 * Two kernel variants:
 *   1. naiveSoftmax  — three separate passes (max, exp-sum, div); readable.
 *   2. fusedSoftmax  — uses shared-memory warp reductions to compute
 *                      max and sum in a single pass, minimising global reads.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -o softmax softmax.cu
 *
 * Run:
 *   ./softmax [rows] [cols]   (default: 4096 4096)
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <functional>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Warp-level reduction helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}


// ---------------------------------------------------------------------------
// Naive softmax: three global passes
// ---------------------------------------------------------------------------
__global__ void rowMaxKernel(const float* x, float* rowMax, int cols) {
    int row = blockIdx.x;
    float maxVal = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        maxVal = fmaxf(maxVal, x[row * cols + j]);
    maxVal = warpReduceMax(maxVal);
    if (threadIdx.x == 0) rowMax[row] = maxVal;
}

__global__ void expSumKernel(const float* x, const float* rowMax,
                              float* rowSum, float* y, int cols) {
    int row = blockIdx.x;
    float m = rowMax[row];
    float s = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float e = expf(x[row * cols + j] - m);
        y[row * cols + j] = e;
        s += e;
    }
    s = warpReduceSum(s);
    if (threadIdx.x == 0) rowSum[row] = s;
}

__global__ void divKernel(float* y, const float* rowSum, int cols) {
    int row = blockIdx.x;
    float s = rowSum[row];
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        y[row * cols + j] /= s;
}


// ---------------------------------------------------------------------------
// Fused softmax: single pass using shared memory
// Each block handles one row; assumes cols <= blockDim.x * 4
// ---------------------------------------------------------------------------
__global__ void fusedSoftmaxKernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // 1. Find row maximum via warp reduction
    float maxVal = -FLT_MAX;
    for (int j = tid; j < cols; j += blockSize)
        maxVal = fmaxf(maxVal, x[row * cols + j]);

    // Reduce max across all threads in block
    smem[tid] = maxVal;
    __syncthreads();
    for (int stride = blockSize / 2; stride > 32; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    // Warp-level final reduce
    if (tid < 32) {
        float val = smem[tid];
        val = warpReduceMax(val);
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();
    maxVal = smem[0];

    // 2. Compute exp(x - max) and partial sum
    float sumVal = 0.f;
    for (int j = tid; j < cols; j += blockSize) {
        float e = expf(x[row * cols + j] - maxVal);
        y[row * cols + j] = e;
        sumVal += e;
    }

    // Reduce sum across block
    smem[tid] = sumVal;
    __syncthreads();
    for (int stride = blockSize / 2; stride > 32; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid < 32) {
        float val = smem[tid];
        val = warpReduceSum(val);
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();
    sumVal = smem[0];

    // 3. Normalize
    for (int j = tid; j < cols; j += blockSize)
        y[row * cols + j] /= sumVal;
}


// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
float timeKernel(std::function<void()> fn, int warmup = 3, int iters = 20) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return ms / iters;
}


// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int rows = (argc > 1) ? atoi(argv[1]) : 4096;
    int cols = (argc > 2) ? atoi(argv[2]) : 4096;

    printf("Row-wise softmax: %d x %d\n", rows, cols);

    size_t sz = (size_t)rows * cols * sizeof(float);
    float *hX = (float*)malloc(sz);
    float *hY_naive = (float*)malloc(sz);
    float *hY_fused = (float*)malloc(sz);

    srand(42);
    for (int i = 0; i < rows * cols; ++i)
        hX[i] = (rand() / (float)RAND_MAX) * 4.f - 2.f;

    float *dX, *dY, *dRowMax, *dRowSum;
    CUDA_CHECK(cudaMalloc(&dX, sz));
    CUDA_CHECK(cudaMalloc(&dY, sz));
    CUDA_CHECK(cudaMalloc(&dRowMax, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRowSum, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, hX, sz, cudaMemcpyHostToDevice));

    int threads = min(cols, 256);

    // --- Naive (3-pass) ---
    auto runNaive = [&]() {
        rowMaxKernel<<<rows, threads>>>(dX, dRowMax, cols);
        expSumKernel<<<rows, threads>>>(dX, dRowMax, dRowSum, dY, cols);
        divKernel<<<rows, threads>>>(dY, dRowSum, cols);
    };
    float naiveMs = timeKernel(runNaive);
    CUDA_CHECK(cudaMemcpy(hY_naive, dY, sz, cudaMemcpyDeviceToHost));

    // --- Fused (1-pass) ---
    size_t sharedMem = threads * sizeof(float);
    auto runFused = [&]() {
        fusedSoftmaxKernel<<<rows, threads, sharedMem>>>(dX, dY, cols);
    };
    float fusedMs = timeKernel(runFused);
    CUDA_CHECK(cudaMemcpy(hY_fused, dY, sz, cudaMemcpyDeviceToHost));

    // Validation
    float maxErr = 0.f;
    for (int i = 0; i < rows * cols; ++i)
        maxErr = fmaxf(maxErr, fabsf(hY_naive[i] - hY_fused[i]));

    printf("\n--- Results ---\n");
    printf("Naive softmax (3-pass): %7.3f ms\n", naiveMs);
    printf("Fused softmax (1-pass): %7.3f ms\n", fusedMs);
    printf("Speedup                : %.2fx\n", naiveMs / fusedMs);
    printf("Max error (naive vs fused): %.4e  %s\n",
           maxErr, maxErr < 1e-5f ? "PASS" : "FAIL");

    double bandwidth = 2.0 * sz / (fusedMs * 1e-3) / 1e9;
    printf("Fused effective bandwidth: %.1f GB/s\n", bandwidth);

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaFree(dRowMax));
    CUDA_CHECK(cudaFree(dRowSum));
    free(hX); free(hY_naive); free(hY_fused);
    return 0;
}
