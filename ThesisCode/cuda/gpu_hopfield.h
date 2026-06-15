/*
 * gpu_hopfield.h - GPU-accelerated Hopfield network operations
 *
 * Drop-in GPU replacements for the computational hotspots in the
 * original thesis code. Uses CUDA for matrix-vector multiply,
 * weight updates, and batch relaxation.
 *
 * Author: Simon McCallum (modernisation 2026)
 */

#ifndef _gpu_hopfield_h_
#define _gpu_hopfield_h_

#ifdef __cplusplus
extern "C" {
#endif

/* Initialise GPU resources. Call once at startup.
 * Returns 0 on success, -1 if no GPU available. */
int gpu_init(int N, int has_bias);

/* Free GPU resources. Call at shutdown. */
void gpu_cleanup(void);

/* Upload weight matrix from host to device.
 * weights: row-major N x (N + bias) array of doubles */
void gpu_upload_weights(const double *weights, int N, int cols);

/* Download weight matrix from device to host. */
void gpu_download_weights(double *weights, int N, int cols);

/* Upload a pattern (unit activations) to device. */
void gpu_upload_pattern(const double *units, int N);

/* GPU-accelerated calcInputs: output[i] = sum_j weight[i][j] * units[j]
 * Replaces the O(N^2) inner loop in net.c:calcInputs().
 * Results are written to unitInput on device and copied back to host. */
void gpu_calc_inputs(double *unitInput, int N, int cols);

/* GPU-accelerated Hebbian weight update.
 * Adds learningConst * outer_product(pattern, pattern) to weights,
 * with optional noise and probability adjustment.
 * cappingType: 0=NOCAP, 1=HARDCAP, 2=SOFTCAP */
void gpu_hebbian_update(const double *units, int N, int cols,
                        double learningConst, int cappingType,
                        double cappingValue);

/* GPU-accelerated delta weight update.
 * weightDelta[i][j] += learningConst * error[i] * input[j] * ratio */
void gpu_delta_weight_update(const double *error, const double *input,
                             int N, int cols, double learningConst,
                             double ratio);

/* Apply weight deltas to weights on device.
 * Also zeroes the diagonal (w[i][i] = 0). */
void gpu_apply_weight_deltas(int N, int cols);

/* Reset weight deltas to zero on device. */
void gpu_reset_weight_deltas(int N, int cols);

/* Batch relaxation: relax multiple patterns in parallel.
 * patterns: array of num_patterns pattern vectors (each N doubles)
 * results:  output array for relaxed patterns
 * cycles:   output array for cycle counts
 * Returns number of patterns that converged. */
int gpu_batch_relax(const double *patterns, double *results, int *cycles,
                    int num_patterns, int N, int cols, int max_cycles,
                    double ON, double OFF);

/* Batch pattern comparison: compute Hamming distances between
 * a test pattern and an array of stored patterns.
 * distances: output array of num_stored integers */
void gpu_batch_hamming(const double *test_pattern,
                       const double *stored_patterns,
                       int *distances, int num_stored, int N);

/* GPU-accelerated basin sampling: run multiple overlap levels in parallel.
 * attractor: the stable pattern to test
 * overlaps:  array of overlap values to test
 * success_rates: output array of success rates
 * num_overlaps: number of overlap levels
 * samples_per: number of random samples per overlap level */
void gpu_batch_basin_sample(const double *attractor,
                            const double *overlaps,
                            double *success_rates,
                            int num_overlaps, int samples_per,
                            int N, int cols, int max_cycles,
                            double ON, double OFF);

#ifdef __cplusplus
}
#endif

#endif /* _gpu_hopfield_h_ */
