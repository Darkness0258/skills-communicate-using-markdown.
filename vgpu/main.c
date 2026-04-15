/**
 * main.c — Virtual GPU demonstration and test suite
 *
 * Runs five benchmarks:
 *   1. Vector addition          – basic element-wise parallelism
 *   2. Vector scale             – scalar broadcast
 *   3. Matrix multiply (tiled)  – 2-D kernel, shared-memory tiles, syncthreads
 *   4. Parallel reduction       – tree reduction with shared memory
 *   5. Histogram                – atomic shared-memory pattern
 *
 * Each benchmark verifies correctness against a serial reference and prints
 * wall-clock timing.
 */

#include "vgpu.h"
#include "kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ─── Timing ────────────────────────────────────────────────────────────── */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ─── Utilities ─────────────────────────────────────────────────────────── */

static void fill_random(float *arr, int n)
{
    for (int i = 0; i < n; i++)
        arr[i] = (float)(rand() % 1000) / 100.0f;
}

static int check_float_array(const float *ref, const float *got,
                              int n, float tol, const char *label)
{
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(ref[i] - got[i]) > tol) {
            if (errors < 5)
                printf("  MISMATCH at [%d]: expected %.6f, got %.6f\n",
                       i, ref[i], got[i]);
            errors++;
        }
    }
    if (errors == 0)
        printf("  [PASS] %s\n", label);
    else
        printf("  [FAIL] %s — %d mismatches\n", label, errors);
    return errors;
}

/* ─── Benchmark 1: Vector addition ─────────────────────────────────────── */

static void bench_vector_add(vgpu_device *dev)
{
    puts("\n══ Benchmark 1: Vector Addition (C = A + B) ══════════════════");

    const int N          = 1 << 16;  /* 64 K elements */
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Host arrays */
    float *h_A = malloc((size_t)N * sizeof(float));
    float *h_B = malloc((size_t)N * sizeof(float));
    float *h_C = malloc((size_t)N * sizeof(float));
    fill_random(h_A, N);
    fill_random(h_B, N);

    /* Device arrays */
    float *d_A = vgpu_malloc((size_t)N * sizeof(float));
    float *d_B = vgpu_malloc((size_t)N * sizeof(float));
    float *d_C = vgpu_malloc((size_t)N * sizeof(float));
    vgpu_memcpy_h2d(d_A, h_A, (size_t)N * sizeof(float));
    vgpu_memcpy_h2d(d_B, h_B, (size_t)N * sizeof(float));

    /* Launch */
    int n = N;
    void *args[] = { d_A, d_B, d_C, &n };
    double t0 = now_sec();
    vgpu_launch(dev, kernel_vector_add,
                dim3_1d(GRID_SIZE), dim3_1d(BLOCK_SIZE), 0, args);
    vgpu_sync(dev);
    double elapsed = now_sec() - t0;

    /* Copy back and verify */
    vgpu_memcpy_d2h(h_C, d_C, (size_t)N * sizeof(float));

    /* Serial reference */
    float *ref = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) ref[i] = h_A[i] + h_B[i];

    check_float_array(ref, h_C, N, 1e-4f, "vector_add");
    printf("  Time : %.3f ms  (%.2f M elements/s)\n",
           elapsed * 1e3, (double)N / elapsed * 1e-6);

    free(h_A); free(h_B); free(h_C); free(ref);
    vgpu_free(d_A); vgpu_free(d_B); vgpu_free(d_C);
}

/* ─── Benchmark 2: Vector scale ─────────────────────────────────────────── */

static void bench_vector_scale(vgpu_device *dev)
{
    puts("\n══ Benchmark 2: Vector Scale (B = alpha × A) ═════════════════");

    const int N          = 1 << 16;
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *h_A = malloc((size_t)N * sizeof(float));
    float *h_B = malloc((size_t)N * sizeof(float));
    fill_random(h_A, N);

    float *d_A = vgpu_malloc((size_t)N * sizeof(float));
    float *d_B = vgpu_malloc((size_t)N * sizeof(float));
    vgpu_memcpy_h2d(d_A, h_A, (size_t)N * sizeof(float));

    float alpha = 3.14f;
    int   n     = N;
    void *args[] = { d_A, d_B, &alpha, &n };
    double t0 = now_sec();
    vgpu_launch(dev, kernel_vector_scale,
                dim3_1d(GRID_SIZE), dim3_1d(BLOCK_SIZE), 0, args);
    vgpu_sync(dev);
    double elapsed = now_sec() - t0;

    vgpu_memcpy_d2h(h_B, d_B, (size_t)N * sizeof(float));

    float *ref = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) ref[i] = alpha * h_A[i];

    check_float_array(ref, h_B, N, 1e-4f, "vector_scale");
    printf("  Time : %.3f ms\n", elapsed * 1e3);

    free(h_A); free(h_B); free(ref);
    vgpu_free(d_A); vgpu_free(d_B);
}

/* ─── Benchmark 3: Tiled matrix multiply ───────────────────────────────── */

#define MATMUL_TILE 16

static void bench_matmul(vgpu_device *dev)
{
    puts("\n══ Benchmark 3: Tiled Matrix Multiply (C = A × B) ═══════════");

    const int N         = 128;   /* 128×128 matrices */
    const int TILE      = MATMUL_TILE;
    const int GRID_SIDE = N / TILE;

    float *h_A = malloc((size_t)N * N * sizeof(float));
    float *h_B = malloc((size_t)N * N * sizeof(float));
    float *h_C = malloc((size_t)N * N * sizeof(float));
    fill_random(h_A, N * N);
    fill_random(h_B, N * N);

    float *d_A = vgpu_malloc((size_t)N * N * sizeof(float));
    float *d_B = vgpu_malloc((size_t)N * N * sizeof(float));
    float *d_C = vgpu_malloc((size_t)N * N * sizeof(float));
    vgpu_memcpy_h2d(d_A, h_A, (size_t)N * N * sizeof(float));
    vgpu_memcpy_h2d(d_B, h_B, (size_t)N * N * sizeof(float));

    int n = N;
    /* shared memory = 2 tiles × TILE × TILE × sizeof(float) */
    size_t smem = 2 * (size_t)TILE * TILE * sizeof(float);
    void *args[] = { d_A, d_B, d_C, &n };

    double t0 = now_sec();
    vgpu_launch(dev, kernel_matmul,
                dim3_2d(GRID_SIDE, GRID_SIDE),
                dim3_2d(TILE, TILE),
                smem, args);
    vgpu_sync(dev);
    double elapsed = now_sec() - t0;

    vgpu_memcpy_d2h(h_C, d_C, (size_t)N * N * sizeof(float));

    /* Serial reference (O(N³) – only feasible for small N) */
    float *ref = calloc((size_t)N * N, sizeof(float));
    for (int r = 0; r < N; r++)
    for (int c = 0; c < N; c++)
    for (int k = 0; k < N; k++)
        ref[r * N + c] += h_A[r * N + k] * h_B[k * N + c];

    check_float_array(ref, h_C, N * N, 0.1f, "matmul (128x128)");
    double gflops = 2.0 * N * N * N / elapsed * 1e-9;
    printf("  Time : %.3f ms  (%.2f GFLOP/s)\n", elapsed * 1e3, gflops);

    free(h_A); free(h_B); free(h_C); free(ref);
    vgpu_free(d_A); vgpu_free(d_B); vgpu_free(d_C);
}

/* ─── Benchmark 4: Parallel reduction ──────────────────────────────────── */

static void bench_reduce(vgpu_device *dev)
{
    puts("\n══ Benchmark 4: Parallel Tree Reduction (sum) ════════════════");

    const int BLOCK_SIZE = 128;
    const int GRID_SIZE  = 256;
    const int N          = BLOCK_SIZE * GRID_SIZE;

    float *h_in      = malloc((size_t)N          * sizeof(float));
    float *h_partial = malloc((size_t)GRID_SIZE  * sizeof(float));
    fill_random(h_in, N);

    float *d_in  = vgpu_malloc((size_t)N         * sizeof(float));
    float *d_out = vgpu_malloc((size_t)GRID_SIZE * sizeof(float));
    vgpu_memcpy_h2d(d_in, h_in, (size_t)N * sizeof(float));

    size_t smem  = (size_t)BLOCK_SIZE * sizeof(float);
    void *args[] = { d_in, d_out };

    double t0 = now_sec();
    vgpu_launch(dev, kernel_reduce_sum,
                dim3_1d(GRID_SIZE), dim3_1d(BLOCK_SIZE), smem, args);
    vgpu_sync(dev);
    double elapsed = now_sec() - t0;

    vgpu_memcpy_d2h(h_partial, d_out, (size_t)GRID_SIZE * sizeof(float));

    /* Accumulate partial sums on host */
    double gpu_sum = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) gpu_sum += h_partial[i];

    /* Serial reference */
    double ref_sum = 0.0;
    for (int i = 0; i < N; i++) ref_sum += h_in[i];

    double rel_err = fabs(gpu_sum - ref_sum) / fabs(ref_sum);
    printf("  Reference sum : %.6f\n",  ref_sum);
    printf("  vGPU sum      : %.6f\n",  gpu_sum);
    printf("  Relative error: %.2e\n",  rel_err);
    if (rel_err < 1e-4)
        printf("  [PASS] reduce_sum\n");
    else
        printf("  [FAIL] reduce_sum (error too large)\n");
    printf("  Time : %.3f ms\n", elapsed * 1e3);

    free(h_in); free(h_partial);
    vgpu_free(d_in); vgpu_free(d_out);
}

/* ─── Benchmark 5: Histogram ────────────────────────────────────────────── */

static void bench_histogram(vgpu_device *dev)
{
    puts("\n══ Benchmark 5: 256-bin Histogram ════════════════════════════");

    const int N          = 1 << 16;
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uint8_t *h_data = malloc((size_t)N * sizeof(uint8_t));
    int      h_hist[256];
    for (int i = 0; i < N; i++) h_data[i] = (uint8_t)(rand() & 0xFF);

    uint8_t *d_data = vgpu_malloc((size_t)N * sizeof(uint8_t));
    int     *d_hist = vgpu_malloc(256 * sizeof(int));
    vgpu_memcpy_h2d(d_data, h_data, (size_t)N * sizeof(uint8_t));
    memset(d_hist, 0, 256 * sizeof(int));  /* zero histogram on "device" */

    int   n    = N;
    void *args[] = { d_data, d_hist, &n };
    size_t smem  = 256 * sizeof(int);

    double t0 = now_sec();
    vgpu_launch(dev, kernel_histogram,
                dim3_1d(GRID_SIZE), dim3_1d(BLOCK_SIZE), smem, args);
    vgpu_sync(dev);
    double elapsed = now_sec() - t0;

    vgpu_memcpy_d2h(h_hist, d_hist, 256 * sizeof(int));

    /* Serial reference */
    int ref_hist[256] = {0};
    for (int i = 0; i < N; i++) ref_hist[h_data[i]]++;

    int errors = 0;
    for (int b = 0; b < 256; b++) {
        if (h_hist[b] != ref_hist[b]) {
            if (errors < 5)
                printf("  MISMATCH bin[%d]: expected %d, got %d\n",
                       b, ref_hist[b], h_hist[b]);
            errors++;
        }
    }
    if (errors == 0)
        printf("  [PASS] histogram\n");
    else
        printf("  [FAIL] histogram — %d bin mismatches\n", errors);
    printf("  Time : %.3f ms\n", elapsed * 1e3);

    free(h_data);
    vgpu_free(d_data); vgpu_free(d_hist);
}

/* ─── Main ──────────────────────────────────────────────────────────────── */

int main(void)
{
    srand(42);

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║          vGPU — Virtual GPU Compute Emulator                 ║\n");
    printf("║  Streaming Multiprocessors backed by POSIX threads           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* 4 SMs × 256 threads/SM ≈ a modest mid-range GPU configuration */
    vgpu_device *dev = vgpu_create(4, 256);

    bench_vector_add  (dev);
    bench_vector_scale(dev);
    bench_matmul      (dev);
    bench_reduce      (dev);
    bench_histogram   (dev);

    vgpu_print_stats(dev);
    vgpu_destroy(dev);

    puts("Done.");
    return 0;
}
