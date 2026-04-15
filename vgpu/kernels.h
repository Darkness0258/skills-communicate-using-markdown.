/**
 * kernels.h — Example compute kernels for the virtual GPU emulator
 *
 * Each kernel follows the vgpu_kernel_fn signature:
 *   void kernel_name(vgpu_thread_ctx *ctx)
 *
 * Arguments are passed via ctx->args[] as a void* array.
 * See kernels.c for argument layouts.
 */

#ifndef KERNELS_H
#define KERNELS_H

#include "vgpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * kernel_vector_add – C[i] = A[i] + B[i]
 *
 * args[0] = float *A   (device)
 * args[1] = float *B   (device)
 * args[2] = float *C   (device, output)
 * args[3] = int   *N   (element count)
 */
void kernel_vector_add(vgpu_thread_ctx *ctx);

/**
 * kernel_vector_scale – B[i] = alpha * A[i]
 *
 * args[0] = float *A     (device, input)
 * args[1] = float *B     (device, output)
 * args[2] = float *alpha (scalar)
 * args[3] = int   *N
 */
void kernel_vector_scale(vgpu_thread_ctx *ctx);

/**
 * kernel_matmul – C = A × B  (row-major, square N×N matrices)
 *
 * args[0] = float *A  (device, N×N)
 * args[1] = float *B  (device, N×N)
 * args[2] = float *C  (device, N×N, output)
 * args[3] = int   *N
 *
 * Launch with a 2-D grid and 2-D block, e.g.
 *   grid  = dim3_2d(N/TILE, N/TILE)
 *   block = dim3_2d(TILE, TILE)
 *
 * Uses shared memory tiles of size TILE×TILE to reduce global memory traffic,
 * just like a real GPU tiled matmul.
 */
void kernel_matmul(vgpu_thread_ctx *ctx);

/**
 * kernel_reduce_sum – parallel tree reduction, producing per-block partial sums.
 *
 * Each block reduces blockDim.x elements into one output value (output[blockIdx.x]).
 * To get the full sum, launch a second reduction pass over the partial sums,
 * or accumulate them on the host.
 *
 * args[0] = float *input   (device, gridDim.x * blockDim.x elements)
 * args[1] = float *output  (device, gridDim.x partial sums)
 *
 * shared_mem_bytes must be >= blockDim.x * sizeof(float)
 *
 * Demonstrates vgpu_syncthreads() for intra-block synchronisation.
 */
void kernel_reduce_sum(vgpu_thread_ctx *ctx);

/**
 * kernel_histogram – 256-bin histogram of uint8 values.
 *
 * Uses per-block private histogram in shared memory to avoid global atomic
 * contention, then merges into the global histogram.
 *
 * args[0] = uint8_t *data     (device)
 * args[1] = int     *hist     (device, 256 bins, zero-initialised before launch)
 * args[2] = int     *N
 *
 * shared_mem_bytes must be >= 256 * sizeof(int)
 */
void kernel_histogram(vgpu_thread_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* KERNELS_H */
