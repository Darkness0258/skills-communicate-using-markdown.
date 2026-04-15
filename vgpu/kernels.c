/**
 * kernels.c — Example compute kernels for the virtual GPU emulator
 *
 * These kernels are written exactly as they would be on a real GPU (CUDA/OpenCL),
 * but execute on CPU threads via the vgpu runtime.
 */

#include "kernels.h"
#include <stdint.h>
#include <stdatomic.h>

/* ─── Vector add ────────────────────────────────────────────────────────── */

/*
 * args: float *A, float *B, float *C, int *N
 * One thread per element; threads beyond N are masked off.
 */
void kernel_vector_add(vgpu_thread_ctx *ctx)
{
    const float *A = (const float *)ctx->args[0];
    const float *B = (const float *)ctx->args[1];
    float       *C = (float       *)ctx->args[2];
    const int    N = *(const int  *)ctx->args[3];

    int i = vgpu_global_id(ctx);
    if (i < N)
        C[i] = A[i] + B[i];
}

/* ─── Vector scale ──────────────────────────────────────────────────────── */

/*
 * args: float *A, float *B, float *alpha, int *N
 */
void kernel_vector_scale(vgpu_thread_ctx *ctx)
{
    const float *A     = (const float *)ctx->args[0];
    float       *B     = (float       *)ctx->args[1];
    const float  alpha = *(const float *)ctx->args[2];
    const int    N     = *(const int  *)ctx->args[3];

    int i = vgpu_global_id(ctx);
    if (i < N)
        B[i] = alpha * A[i];
}

/* ─── Tiled matrix multiply ─────────────────────────────────────────────── */

/*
 * Tiled (shared-memory) N×N square matrix multiply C = A × B.
 *
 * TILE must divide N evenly.  shared_mem_bytes = 2 * TILE * TILE * sizeof(float).
 *
 * args: float *A, float *B, float *C, int *N
 */
#define MATMUL_TILE 16

void kernel_matmul(vgpu_thread_ctx *ctx)
{
    const float *A = (const float *)ctx->args[0];
    const float *B = (const float *)ctx->args[1];
    float       *C = (float       *)ctx->args[2];
    const int    N = *(const int  *)ctx->args[3];

    const int tx = ctx->threadIdx.x;
    const int ty = ctx->threadIdx.y;
    const int row = ctx->blockIdx.y * MATMUL_TILE + ty;
    const int col = ctx->blockIdx.x * MATMUL_TILE + tx;

    /* Two TILE×TILE tiles in shared memory */
    float (*sA)[MATMUL_TILE] = (float (*)[MATMUL_TILE])ctx->shared_mem;
    float (*sB)[MATMUL_TILE] = sA + MATMUL_TILE;   /* second tile follows sA */

    float acc = 0.0f;

    for (int tile = 0; tile < N / MATMUL_TILE; tile++) {
        /* Load tiles collaboratively */
        sA[ty][tx] = (row < N && tile * MATMUL_TILE + tx < N)
                     ? A[row * N + tile * MATMUL_TILE + tx] : 0.0f;
        sB[ty][tx] = (col < N && tile * MATMUL_TILE + ty < N)
                     ? B[(tile * MATMUL_TILE + ty) * N + col] : 0.0f;

        vgpu_syncthreads(ctx);  /* wait until both tiles are ready */

        for (int k = 0; k < MATMUL_TILE; k++)
            acc += sA[ty][k] * sB[k][tx];

        vgpu_syncthreads(ctx);  /* wait before overwriting tiles */
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

/* ─── Parallel tree reduction (sum) ────────────────────────────────────── */

/*
 * Classic GPU reduction.  Each block produces one partial sum stored in
 * output[blockIdx.x].
 *
 * args: float *input, float *output
 * shared_mem_bytes >= blockDim.x * sizeof(float)
 */
void kernel_reduce_sum(vgpu_thread_ctx *ctx)
{
    const float *input  = (const float *)ctx->args[0];
    float       *output = (float       *)ctx->args[1];
    float       *sdata  = (float        *)ctx->shared_mem;

    int tid = vgpu_local_id(ctx);
    int gid = vgpu_global_id(ctx);

    /* Load one element per thread into shared memory */
    sdata[tid] = input[gid];
    vgpu_syncthreads(ctx);

    /* Tree reduction: stride halves each iteration */
    for (int s = ctx->blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        vgpu_syncthreads(ctx);
    }

    /* Thread 0 writes the block result */
    if (tid == 0)
        output[ctx->blockIdx.x] = sdata[0];
}

/* ─── 256-bin histogram ─────────────────────────────────────────────────── */

/*
 * Each block accumulates a private histogram in shared memory, then merges
 * it into the global histogram with atomic adds.
 *
 * args: uint8_t *data, int *hist, int *N
 * shared_mem_bytes >= 256 * sizeof(int)
 */
void kernel_histogram(vgpu_thread_ctx *ctx)
{
    const uint8_t *data  = (const uint8_t *)ctx->args[0];
    int           *hist  = (int            *)ctx->args[1];
    const int      N     = *(const int     *)ctx->args[2];

    /* Per-block private histogram lives in shared memory */
    int *local_hist = (int *)ctx->shared_mem;

    int tid = vgpu_local_id(ctx);
    int bsz = ctx->blockDim.x * ctx->blockDim.y * ctx->blockDim.z;

    /* Initialise private histogram (first 256 threads each clear one bin) */
    if (tid < 256)
        local_hist[tid] = 0;
    vgpu_syncthreads(ctx);

    /* Each thread processes its element */
    int gid = vgpu_global_id(ctx);
    if (gid < N) {
        int bin = data[gid];
        /* Atomic increment of the local (shared memory) bin.
         * Using C11 _Atomic; in a real GPU this would be an atomic intrinsic. */
        (void)bsz;  /* suppress unused warning */
        __atomic_fetch_add(&local_hist[bin], 1, __ATOMIC_RELAXED);
    }

    vgpu_syncthreads(ctx);

    /* Merge local histogram into global histogram (first 256 threads) */
    if (tid < 256)
        __atomic_fetch_add(&hist[tid], local_hist[tid], __ATOMIC_RELAXED);
}
