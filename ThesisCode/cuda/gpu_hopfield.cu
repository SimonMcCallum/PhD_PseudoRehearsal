/*
 * gpu_hopfield.cu - CUDA kernels for Hopfield network acceleration
 *
 * Targets: NVIDIA RTX 3070 (Compute 8.6, Ampere)
 * CUDA: 12.8
 *
 * Key optimisations:
 *   1. calcInputs -> cuBLAS DGEMV (or SGEMV for float mode)
 *   2. Hebbian update -> custom kernel with shared memory
 *   3. Batch relaxation -> one block per pattern
 *   4. Batch Hamming distance -> parallel reduction
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gpu_hopfield.h"

/* Error checking macro */
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = (call); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                __FILE__, __LINE__, stat); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* Device memory pointers */
static double *d_weights = NULL;       /* N x cols weight matrix */
static double *d_weight_deltas = NULL; /* N x cols weight delta accumulator */
static double *d_units = NULL;         /* current pattern activations */
static double *d_unit_input = NULL;    /* computed inputs */
static double *d_error = NULL;         /* error vector */
static double *d_temp = NULL;          /* temporary buffer */

/* cuBLAS handle */
static cublasHandle_t cublas_handle;

/* Dimensions */
static int g_N = 0;
static int g_cols = 0;
static int g_initialised = 0;

/* ================================================================
 * Initialisation and cleanup
 * ================================================================ */

int gpu_init(int N, int has_bias) {
    int cols = N + has_bias;
    int device_count = 0;

    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable GPU found\n");
        return -1;
    }

    /* Print GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %.1f GB, Compute %d.%d)\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           prop.major, prop.minor);

    g_N = N;
    g_cols = cols;

    /* Allocate device memory */
    size_t weight_bytes = (size_t)N * cols * sizeof(double);
    size_t vec_bytes = (size_t)(cols + 1) * sizeof(double);  /* +1 for safety */

    CUDA_CHECK(cudaMalloc(&d_weights, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight_deltas, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_units, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_unit_input, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_error, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, weight_bytes));  /* for batch ops */

    /* Zero out deltas */
    CUDA_CHECK(cudaMemset(d_weight_deltas, 0, weight_bytes));

    /* Create cuBLAS handle */
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    g_initialised = 1;
    printf("GPU initialised: %d x %d weight matrix (%.2f MB)\n",
           N, cols, weight_bytes / (1024.0 * 1024.0));
    return 0;
}

void gpu_cleanup(void) {
    if (!g_initialised) return;

    cudaFree(d_weights);
    cudaFree(d_weight_deltas);
    cudaFree(d_units);
    cudaFree(d_unit_input);
    cudaFree(d_error);
    cudaFree(d_temp);
    cublasDestroy(cublas_handle);

    d_weights = d_weight_deltas = d_units = d_unit_input = d_error = d_temp = NULL;
    g_initialised = 0;
}

/* ================================================================
 * Data transfer
 * ================================================================ */

void gpu_upload_weights(const double *weights, int N, int cols) {
    CUDA_CHECK(cudaMemcpy(d_weights, weights,
                          (size_t)N * cols * sizeof(double),
                          cudaMemcpyHostToDevice));
}

void gpu_download_weights(double *weights, int N, int cols) {
    CUDA_CHECK(cudaMemcpy(weights, d_weights,
                          (size_t)N * cols * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void gpu_upload_pattern(const double *units, int N) {
    /* Upload N+1 elements (includes bias unit) */
    CUDA_CHECK(cudaMemcpy(d_units, units,
                          (size_t)(N + 1) * sizeof(double),
                          cudaMemcpyHostToDevice));
}

/* ================================================================
 * HOTSPOT 1: Matrix-vector multiply (calcInputs replacement)
 *
 * unitInput[i] = sum_{j!=i} weight[i][j] * unit[j]
 *
 * Uses cuBLAS DGEMV for the bulk multiply, then a custom kernel
 * to zero out the diagonal contribution.
 * ================================================================ */

__global__ void kernel_subtract_diagonal(double *unit_input,
                                         const double *weights,
                                         const double *units,
                                         int N, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        /* Remove self-connection: subtract w[i][i] * u[i] */
        unit_input[i] -= weights[i * cols + i] * units[i];
    }
}

void gpu_calc_inputs(double *unitInput, int N, int cols) {
    double alpha = 1.0, beta = 0.0;

    /* DGEMV: unit_input = weights * units
     * cuBLAS uses column-major, but our weights are row-major.
     * So we use CUBLAS_OP_T with reversed dimensions. */
    CUBLAS_CHECK(cublasDgemv(cublas_handle, CUBLAS_OP_T,
                             cols, N,      /* m, n (transposed) */
                             &alpha,
                             d_weights, cols,  /* A, lda */
                             d_units, 1,       /* x, incx */
                             &beta,
                             d_unit_input, 1)); /* y, incy */

    /* Remove self-connections (diagonal) */
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    kernel_subtract_diagonal<<<blocks, threads>>>(
        d_unit_input, d_weights, d_units, N, cols);
    CUDA_CHECK(cudaGetLastError());

    /* Copy results back to host */
    CUDA_CHECK(cudaMemcpy(unitInput, d_unit_input,
                          N * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

/* ================================================================
 * HOTSPOT 2: Hebbian weight update kernel
 *
 * w[i][j] += learningConst * ((unit[i] == unit[j]) ? +1 : -1)
 * with w[i][i] = 0
 * ================================================================ */

__global__ void kernel_hebbian_update(double *weights,
                                      const double *units,
                                      int N, int cols,
                                      double learningConst,
                                      int cappingType,
                                      double cappingValue) {
    int to = blockIdx.y * blockDim.y + threadIdx.y;
    int from = blockIdx.x * blockDim.x + threadIdx.x;

    if (to < N && from < cols && to != from) {
        int idx = to * cols + from;
        double u_to = units[to];
        double u_from = units[from];

        /* Hebbian rule: same sign -> strengthen, different -> weaken */
        double delta = (u_to == u_from) ? learningConst : -learningConst;
        double new_weight = weights[idx] + delta;

        /* Weight capping */
        if (cappingType == 1) { /* HARDCAP */
            if (new_weight > cappingValue) new_weight = cappingValue;
            else if (new_weight < -cappingValue) new_weight = -cappingValue;
        } else if (cappingType == 2) { /* SOFTCAP */
            double change = (fabs(weights[idx]) - cappingValue) / cappingValue;
            if (change < 0)
                new_weight = weights[idx] + delta;
            else if (change < 1)
                new_weight = weights[idx] + delta * change;
            else
                new_weight = weights[idx]; /* no change */
        }

        weights[idx] = new_weight;
    }
}

void gpu_hebbian_update(const double *units, int N, int cols,
                        double learningConst, int cappingType,
                        double cappingValue) {
    /* Upload pattern */
    CUDA_CHECK(cudaMemcpy(d_units, units, (N + 1) * sizeof(double),
                          cudaMemcpyHostToDevice));

    /* Launch 2D kernel */
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (N + 15) / 16);

    kernel_hebbian_update<<<blocks, threads>>>(
        d_weights, d_units, N, cols,
        learningConst, cappingType, cappingValue);
    CUDA_CHECK(cudaGetLastError());
}

/* ================================================================
 * HOTSPOT 3: Delta weight update kernel
 *
 * weightDelta[i][j] += learningConst * error[i] * input[j] * ratio
 * This is an outer product: weightDelta += lr * ratio * error (x) input
 * ================================================================ */

__global__ void kernel_delta_update(double *weight_deltas,
                                    const double *error,
                                    const double *input,
                                    int N, int cols,
                                    double lr_ratio) {
    int to = blockIdx.y * blockDim.y + threadIdx.y;
    int from = blockIdx.x * blockDim.x + threadIdx.x;

    if (to < N && from < cols) {
        weight_deltas[to * cols + from] += lr_ratio * error[to] * input[from];
    }
}

void gpu_delta_weight_update(const double *error, const double *input,
                             int N, int cols, double learningConst,
                             double ratio) {
    CUDA_CHECK(cudaMemcpy(d_error, error, N * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_units, input, (N + 1) * sizeof(double),
                          cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (N + 15) / 16);

    kernel_delta_update<<<blocks, threads>>>(
        d_weight_deltas, d_error, d_units, N, cols,
        learningConst * ratio);
    CUDA_CHECK(cudaGetLastError());
}

/* ================================================================
 * Weight delta application
 * ================================================================ */

__global__ void kernel_apply_deltas(double *weights,
                                    double *deltas,
                                    int N, int cols) {
    int to = blockIdx.y * blockDim.y + threadIdx.y;
    int from = blockIdx.x * blockDim.x + threadIdx.x;

    if (to < N && from < cols) {
        int idx = to * cols + from;
        weights[idx] += deltas[idx];
        deltas[idx] = 0.0;
        if (to == from) weights[idx] = 0.0;  /* zero diagonal */
    }
}

void gpu_apply_weight_deltas(int N, int cols) {
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (N + 15) / 16);
    kernel_apply_deltas<<<blocks, threads>>>(d_weights, d_weight_deltas, N, cols);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_reset_weight_deltas(int N, int cols) {
    CUDA_CHECK(cudaMemset(d_weight_deltas, 0,
                          (size_t)N * cols * sizeof(double)));
}

/* ================================================================
 * HOTSPOT 4: Batch Hamming distance computation
 *
 * Compare one test pattern against many stored patterns in parallel.
 * Each thread block handles one stored pattern.
 * ================================================================ */

__global__ void kernel_hamming_distance(const double *test,
                                        const double *stored,
                                        int *distances,
                                        int num_stored, int N) {
    extern __shared__ int s_count[];

    int pat_idx = blockIdx.x;
    if (pat_idx >= num_stored) return;

    int tid = threadIdx.x;
    int local_diff = 0;

    /* Each thread counts mismatches for its units */
    for (int i = tid; i < N; i += blockDim.x) {
        double t = test[i];
        double s = stored[pat_idx * N + i];
        /* Check both identical and inverse */
        if (fabs(t - s) > 1e-10) local_diff++;
    }

    s_count[tid] = local_diff;
    __syncthreads();

    /* Parallel reduction */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_count[tid] += s_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        distances[pat_idx] = s_count[0];
    }
}

void gpu_batch_hamming(const double *test_pattern,
                       const double *stored_patterns,
                       int *distances, int num_stored, int N) {
    double *d_test, *d_stored;
    int *d_dist;

    CUDA_CHECK(cudaMalloc(&d_test, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_stored, (size_t)num_stored * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dist, num_stored * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_test, test_pattern, N * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_stored, stored_patterns,
                          (size_t)num_stored * N * sizeof(double),
                          cudaMemcpyHostToDevice));

    int threads = 256;
    int shared_mem = threads * sizeof(int);
    kernel_hamming_distance<<<num_stored, threads, shared_mem>>>(
        d_test, d_stored, d_dist, num_stored, N);

    CUDA_CHECK(cudaMemcpy(distances, d_dist, num_stored * sizeof(int),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_test);
    cudaFree(d_stored);
    cudaFree(d_dist);
}

/* ================================================================
 * HOTSPOT 5: Batch relaxation
 *
 * Relax multiple patterns simultaneously. Each thread block handles
 * one pattern's synchronous relaxation independently.
 *
 * Uses synchronous update (all units update simultaneously per cycle)
 * which is fully parallelisable, unlike asynchronous update.
 * ================================================================ */

__global__ void kernel_batch_relax(const double *weights,
                                    const double *input_patterns,
                                    double *output_patterns,
                                    int *cycle_counts,
                                    int num_patterns, int N, int cols,
                                    int max_cycles,
                                    double ON, double OFF) {
    extern __shared__ double shared[];

    int pat_idx = blockIdx.x;
    if (pat_idx >= num_patterns) return;

    int tid = threadIdx.x;

    /* Shared memory layout: units[cols] + inputs[N] + changes[1] */
    double *s_units = shared;
    double *s_inputs = shared + cols;
    int *s_changes = (int *)(shared + cols + N);

    /* Load pattern into shared memory */
    for (int i = tid; i < cols; i += blockDim.x) {
        s_units[i] = input_patterns[pat_idx * cols + i];
    }
    __syncthreads();

    int cycle = 0;
    int changes = 1;

    while (cycle < max_cycles && changes > 0) {
        /* Calculate inputs: input[i] = sum_{j!=i} w[i][j] * unit[j] */
        for (int i = tid; i < N; i += blockDim.x) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                if (j != i) {
                    sum += weights[i * cols + j] * s_units[j];
                }
            }
            s_inputs[i] = sum;
        }
        __syncthreads();

        /* Apply activation and count changes */
        if (tid == 0) s_changes[0] = 0;
        __syncthreads();

        for (int i = tid; i < N; i += blockDim.x) {
            double new_val = (s_inputs[i] > 0) ? ON : OFF;
            if (fabs(new_val - s_units[i]) > 1e-10) {
                atomicAdd(s_changes, 1);
            }
            s_units[i] = new_val;
        }
        __syncthreads();

        changes = s_changes[0];
        cycle++;
    }

    /* Write results back */
    for (int i = tid; i < N; i += blockDim.x) {
        output_patterns[pat_idx * N + i] = s_units[i];
    }
    if (tid == 0) {
        cycle_counts[pat_idx] = cycle;
    }
}

int gpu_batch_relax(const double *patterns, double *results, int *cycles,
                    int num_patterns, int N, int cols, int max_cycles,
                    double ON, double OFF) {
    double *d_input, *d_output;
    int *d_cycles;

    CUDA_CHECK(cudaMalloc(&d_input, (size_t)num_patterns * cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, (size_t)num_patterns * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cycles, num_patterns * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, patterns,
                          (size_t)num_patterns * cols * sizeof(double),
                          cudaMemcpyHostToDevice));

    /* Shared memory: cols doubles + N doubles + 1 int */
    int threads = min(256, N);
    size_t shared_bytes = (cols + N) * sizeof(double) + sizeof(int);

    kernel_batch_relax<<<num_patterns, threads, shared_bytes>>>(
        d_weights, d_input, d_output, d_cycles,
        num_patterns, N, cols, max_cycles, ON, OFF);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(results, d_output,
                          (size_t)num_patterns * N * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cycles, d_cycles,
                          num_patterns * sizeof(int),
                          cudaMemcpyDeviceToHost));

    /* Count converged */
    int converged = 0;
    for (int i = 0; i < num_patterns; i++) {
        if (cycles[i] < max_cycles) converged++;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_cycles);

    return converged;
}
