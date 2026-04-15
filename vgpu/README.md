# vGPU — Virtual GPU Compute Emulator

A fully functional software GPU emulator that runs on **any CPU** — no GPU required. Implements the CUDA programming model using POSIX threads.

## Architecture

```
vgpu_launch()
     │
     ▼  FIFO work queue
┌──────────────────────────────────┐
│         Block Work Queue          │
└────────┬─────────┬────────┬──────┘
         │ SM 0    │ SM 1   │ SM n    ← num_sms dispatcher threads
    execute_block() × num_sms
         │
         ▼  enqueue one task per logical GPU thread
┌────────────────────────────────────────┐
│           Thread Task Queue             │
└─┬───┬───┬───┬───┬───┬───┬───┬────────┘
  ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
pool[0] ... pool[num_sms × threads_per_sm - 1]
           (persistent worker threads — created once, reused forever)
```

### Key Design Decisions

| GPU concept        | vGPU implementation                          |
|--------------------|----------------------------------------------|
| Streaming Multiprocessor (SM) | pthread dispatcher + per-SM work queue |
| Thread block        | Group of logical threads with shared memory |
| Shared memory       | `calloc`-allocated arena, per-block lifetime |
| `__syncthreads()`  | `pthread_barrier_t` initialised with `block_size` |
| Thread pool         | `num_sms × threads_per_sm` persistent pthreads |
| `cudaMalloc`        | `malloc` (unified address space emulation)   |
| `cudaMemcpy`        | `memcpy`                                     |

## API

```c
/* Device lifecycle */
vgpu_device *vgpu_create(int num_sms, int threads_per_sm);
void         vgpu_destroy(vgpu_device *dev);

/* Memory */
void *vgpu_malloc(size_t size);
void  vgpu_free(void *ptr);
void  vgpu_memcpy_h2d(void *dst, const void *src, size_t n);
void  vgpu_memcpy_d2h(void *dst, const void *src, size_t n);

/* Kernel launch */
void vgpu_launch(vgpu_device *dev, vgpu_kernel_fn kernel,
                 vgpu_dim3 grid, vgpu_dim3 block,
                 size_t shared_mem_bytes, void **args);
void vgpu_sync(vgpu_device *dev);

/* Inside a kernel */
void vgpu_syncthreads(vgpu_thread_ctx *ctx);  /* like __syncthreads() */
int  vgpu_global_id(const vgpu_thread_ctx *ctx);
int  vgpu_local_id (const vgpu_thread_ctx *ctx);
```

### Writing a Kernel

```c
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
```

### Launching a Kernel

```c
vgpu_device *dev = vgpu_create(4, 256);   // 4 SMs, 256 threads/SM

int   N = 65536;
float *d_A = vgpu_malloc(N * sizeof(float));
float *d_B = vgpu_malloc(N * sizeof(float));
float *d_C = vgpu_malloc(N * sizeof(float));

// ... fill d_A and d_B via vgpu_memcpy_h2d ...

void *args[] = { d_A, d_B, d_C, &N };
vgpu_launch(dev, kernel_vector_add,
            dim3_1d((N + 255) / 256),   // grid: one block per 256 elements
            dim3_1d(256),               // block: 256 threads
            0, args);                   // 0 bytes of shared memory
vgpu_sync(dev);

vgpu_memcpy_d2h(result, d_C, N * sizeof(float));
vgpu_destroy(dev);
```

## Included Kernels (`kernels.c`)

| Kernel | Description | Concepts demonstrated |
|--------|-------------|----------------------|
| `kernel_vector_add`   | `C[i] = A[i] + B[i]` | 1-D grid, global thread ID |
| `kernel_vector_scale` | `B[i] = alpha * A[i]` | scalar broadcast |
| `kernel_matmul`       | Tiled N×N matrix multiply | 2-D grid, shared memory tiles, `vgpu_syncthreads` |
| `kernel_reduce_sum`   | Parallel tree reduction | shared memory, logarithmic reduction, `vgpu_syncthreads` |
| `kernel_histogram`    | 256-bin histogram | atomic ops in shared memory, merge pattern |

## Build & Run

```bash
make          # build
make run      # build + run all benchmarks
make clean    # clean build artefacts
```

### Requirements

- GCC ≥ 7 with C11 support  
- POSIX threads (`-lpthread`)
- `libm` (`-lm`)

### Sample Output

```
[vGPU] Device created: 4 SMs x 256 threads/SM  (thread pool = 1024)

== Benchmark 1: Vector Addition (C = A + B) ==
  [PASS] vector_add
  Time : 152 ms

== Benchmark 3: Tiled Matrix Multiply (C = A x B) ==
  [PASS] matmul (128x128)
  Time : 837 ms  (0.01 GFLOP/s)

== Benchmark 4: Parallel Tree Reduction (sum) ==
  Reference sum : 163449.059957
  vGPU sum      : 163449.059570
  Relative error: 2.37e-09
  [PASS] reduce_sum

[vGPU] -- Execution Statistics --
  Thread pool size          : 1024
  Blocks  executed          : 1088
  Threads executed          : 245760
```

## Limitations & Differences from a Real GPU

| Feature | Real GPU | vGPU |
|---------|----------|------|
| Parallelism | Hardware SIMD warps | OS threads (context-switched) |
| Shared memory | On-chip SRAM (fast) | Heap allocation (cache-speed) |
| Thread creation | Zero cost (persistent hardware) | One-time per device lifetime |
| Memory bandwidth | High-bandwidth HBM/GDDR | System DRAM |
| Occupancy | Hardware scheduler | Bounded by pool_size ≥ block_size |
| Warp divergence | Performance loss | No effect (all branches taken) |

## Extending the Emulator

- **Add warp-level primitives** (`__shfl_sync`, `__ballot_sync`) using compiler intrinsics.
- **Instruction-level tracing**: wrap `vgpu_kernel_fn` with a trampoline that records every call.
- **SIMD vectorisation**: batch warps of 8 or 16 logical threads to exploit AVX2/AVX-512.
- **Multi-device**: create multiple `vgpu_device` instances and migrate memory between them.
