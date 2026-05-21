/*
 * Tiled matrix multiplication in CUDA C++.
 *
 * Implements: C = A @ B  where A is (M x K), B is (K x N), C is (M x N).
 *
 * Key techniques:
 *   - Shared-memory tiling: each block loads TILE_SIZE x TILE_SIZE sub-tiles
 *     of A and B into shared memory, reducing global-memory reads by TILE_SIZE.
 *   - __syncthreads() barriers prevent read-before-write hazards between tiles.
 *   - Boundary checking handles non-multiple-of-TILE_SIZE matrix dimensions.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -o matmul matmul.cu
 *
 * Run:
 *   ./matmul [M] [N] [K]   (default: 1024 1024 1024)
 */

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// ---------------------------------------------------------------------------
// Error-check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


// ---------------------------------------------------------------------------
// Tiled GEMM kernel
// ---------------------------------------------------------------------------
__global__ void tiledMatmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K
) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}


// ---------------------------------------------------------------------------
// Naive reference kernel (for correctness comparison)
// ---------------------------------------------------------------------------
__global__ void naiveMatmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}


// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
void cpuMatmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}


// ---------------------------------------------------------------------------
// Timing helper using CUDA events
// ---------------------------------------------------------------------------
float timeKernel(std::function<void()> fn, int warmup = 3, int iters = 10) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}


// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("Matrix multiply: (%d x %d) @ (%d x %d)\n", M, K, K, N);

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    // Host allocations
    float *hA = (float*)malloc(sizeA);
    float *hB = (float*)malloc(sizeB);
    float *hC_ref = (float*)malloc(sizeC);
    float *hC_tiled = (float*)malloc(sizeC);

    // Initialize with random values
    srand(42);
    for (int i = 0; i < M * K; ++i) hA[i] = (rand() / (float)RAND_MAX) - 0.5f;
    for (int i = 0; i < K * N; ++i) hB[i] = (rand() / (float)RAND_MAX) - 0.5f;

    // CPU reference (small sizes only)
    if ((long long)M * N * K <= 64 * 1024 * 1024) {
        printf("Computing CPU reference...\n");
        cpuMatmul(hA, hB, hC_ref, M, N, K);
    }

    // Device allocations
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeA));
    CUDA_CHECK(cudaMalloc(&dB, sizeB));
    CUDA_CHECK(cudaMalloc(&dC, sizeC));

    CUDA_CHECK(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Tiled kernel benchmark
    auto runTiled = [&]() {
        tiledMatmul<<<grid, block>>>(dA, dB, dC, M, N, K);
    };
    float tiledMs = timeKernel(runTiled);
    CUDA_CHECK(cudaMemcpy(hC_tiled, dC, sizeC, cudaMemcpyDeviceToHost));

    // Naive kernel benchmark
    auto runNaive = [&]() {
        naiveMatmul<<<grid, block>>>(dA, dB, dC, M, N, K);
    };
    float naiveMs = timeKernel(runNaive);

    // TFLOPS
    double flops = 2.0 * M * N * K;
    double tiledTflops = flops / (tiledMs * 1e-3) / 1e12;
    double naiveTflops = flops / (naiveMs * 1e-3) / 1e12;

    printf("\n--- Results ---\n");
    printf("Tiled matmul : %7.3f ms   %.2f TFLOPS\n", tiledMs, tiledTflops);
    printf("Naive matmul : %7.3f ms   %.2f TFLOPS\n", naiveMs, naiveTflops);
    printf("Speedup      : %.2fx\n", naiveMs / tiledMs);

    // Validation against CPU (if available)
    if ((long long)M * N * K <= 64 * 1024 * 1024) {
        float maxErr = 0.f;
        for (int i = 0; i < M * N; ++i)
            maxErr = fmaxf(maxErr, fabsf(hC_ref[i] - hC_tiled[i]));
        printf("Max error vs CPU: %.4e  %s\n", maxErr, maxErr < 1e-2f ? "PASS" : "FAIL");
    }

    // GPU memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    printf("GPU memory: %.0f MB used / %.0f MB total\n",
           (totalMem - freeMem) / 1024.0 / 1024.0,
           totalMem / 1024.0 / 1024.0);

    // Cleanup
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC_ref); free(hC_tiled);

    return 0;
}
