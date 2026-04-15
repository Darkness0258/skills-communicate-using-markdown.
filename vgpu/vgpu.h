/**
 * vgpu.h — Virtual GPU Compute Emulator
 *
 * A CUDA-inspired API that executes kernels on CPU threads, modelling:
 *   - Streaming Multiprocessors (SMs) as a thread-pool that drains a work queue
 *   - Thread blocks as parallel pthreads with a per-block shared memory arena
 *   - __syncthreads() via pthread_barrier_t
 *
 * Quick-start
 * -----------
 *   vgpu_device *dev = vgpu_create(4, 256);   // 4 SMs, 256 threads/SM
 *
 *   float *d_a = vgpu_malloc(N * sizeof(float));
 *   vgpu_memcpy_h2d(d_a, h_a, N * sizeof(float));
 *
 *   void *args[] = { d_a, d_b, d_c, &N };
 *   vgpu_launch(dev, my_kernel,
 *               dim3_1d((N + 255) / 256),  // grid
 *               dim3_1d(256),              // block
 *               0, args);
 *   vgpu_sync(dev);
 *
 *   vgpu_memcpy_d2h(h_c, d_c, N * sizeof(float));
 *   vgpu_free(d_a);
 *   vgpu_destroy(dev);
 */

#ifndef VGPU_H
#define VGPU_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── 3-D dimension ─────────────────────────────────────────────────────── */

typedef struct { int x, y, z; } vgpu_dim3;

#define make_dim3(X, Y, Z) ((vgpu_dim3){(X), (Y), (Z)})
#define dim3_1d(X)         make_dim3((X), 1, 1)
#define dim3_2d(X, Y)      make_dim3((X), (Y), 1)

/* ─── Thread context (passed to every kernel invocation) ────────────────── */

typedef struct {
    vgpu_dim3          threadIdx;   /* position of this thread within its block */
    vgpu_dim3          blockIdx;    /* position of the block within the grid     */
    vgpu_dim3          blockDim;    /* block dimensions                          */
    vgpu_dim3          gridDim;     /* grid dimensions                           */
    void              *shared_mem;  /* per-block shared memory arena             */
    void             **args;        /* kernel argument array                     */
    pthread_barrier_t *barrier;     /* use vgpu_syncthreads() – do not touch     */
} vgpu_thread_ctx;

/* ─── Kernel type ───────────────────────────────────────────────────────── */

typedef void (*vgpu_kernel_fn)(vgpu_thread_ctx *ctx);

/* ─── Device handle ─────────────────────────────────────────────────────── */

typedef struct vgpu_device vgpu_device;

/* ─── Lifecycle ─────────────────────────────────────────────────────────── */

/**
 * vgpu_create – allocate a virtual GPU.
 *
 * @param num_sms        Number of Streaming Multiprocessors (SM worker threads).
 *                       Each SM drains blocks from a shared work queue.
 * @param threads_per_sm Informational; controls how many blocks run concurrently
 *                       per SM (each SM occupies at most this many OS threads at
 *                       a time for block-level work).
 */
vgpu_device *vgpu_create(int num_sms, int threads_per_sm);

/** vgpu_destroy – synchronise and tear down all SM threads. */
void vgpu_destroy(vgpu_device *dev);

/* ─── Memory ────────────────────────────────────────────────────────────── */

/** vgpu_malloc – allocate device memory (plain heap in this emulator). */
void *vgpu_malloc(size_t size);

/** vgpu_free – release device memory. */
void  vgpu_free(void *ptr);

/** vgpu_memcpy_h2d – host → device copy. */
void  vgpu_memcpy_h2d(void *dst, const void *src, size_t n);

/** vgpu_memcpy_d2h – device → host copy. */
void  vgpu_memcpy_d2h(void *dst, const void *src, size_t n);

/* ─── Kernel launch ─────────────────────────────────────────────────────── */

/**
 * vgpu_launch – enqueue a kernel for execution.
 *
 * All blocks are enqueued atomically into the SM work queue and processed
 * by SM worker threads as they become available.
 *
 * @param dev              The virtual GPU device.
 * @param kernel           Kernel function pointer.
 * @param grid_dim         Number of blocks in each dimension.
 * @param block_dim        Number of threads per block in each dimension.
 * @param shared_mem_bytes Bytes of per-block shared memory (zero-initialised).
 * @param args             Pointer array of arguments passed to the kernel via
 *                         ctx->args[].  The length and types of each element are
 *                         kernel-defined; the array must remain valid until
 *                         vgpu_sync() returns.
 */
void vgpu_launch(vgpu_device    *dev,
                 vgpu_kernel_fn  kernel,
                 vgpu_dim3       grid_dim,
                 vgpu_dim3       block_dim,
                 size_t          shared_mem_bytes,
                 void          **args);

/** vgpu_sync – block until all previously launched kernels complete. */
void vgpu_sync(vgpu_device *dev);

/* ─── In-kernel primitive ───────────────────────────────────────────────── */

/**
 * vgpu_syncthreads – synchronise all threads in a block.
 *
 * Equivalent to __syncthreads() in CUDA.  Every thread in the block must
 * call this before any thread is allowed to continue past it.
 */
static inline void vgpu_syncthreads(vgpu_thread_ctx *ctx)
{
    pthread_barrier_wait(ctx->barrier);
}

/* ─── Helpers ───────────────────────────────────────────────────────────── */

/** Global linear thread ID across the whole grid. */
static inline int vgpu_global_id(const vgpu_thread_ctx *ctx)
{
    int block_id =  ctx->blockIdx.z  * ctx->gridDim.x  * ctx->gridDim.y
                  + ctx->blockIdx.y  * ctx->gridDim.x
                  + ctx->blockIdx.x;
    int thread_id = ctx->threadIdx.z * ctx->blockDim.x * ctx->blockDim.y
                  + ctx->threadIdx.y * ctx->blockDim.x
                  + ctx->threadIdx.x;
    int block_size = ctx->blockDim.x * ctx->blockDim.y * ctx->blockDim.z;
    return block_id * block_size + thread_id;
}

/** Linear thread ID within its block. */
static inline int vgpu_local_id(const vgpu_thread_ctx *ctx)
{
    return ctx->threadIdx.z * ctx->blockDim.x * ctx->blockDim.y
         + ctx->threadIdx.y * ctx->blockDim.x
         + ctx->threadIdx.x;
}

/* ─── Diagnostics ───────────────────────────────────────────────────────── */

/** Print device configuration and execution statistics to stdout. */
void vgpu_print_stats(vgpu_device *dev);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_H */
