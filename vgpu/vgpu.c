/**
 * vgpu.c — Virtual GPU Compute Emulator — core runtime
 *
 * Architecture (persistent thread pool edition)
 * ─────────────────────────────────────────────
 *
 *  vgpu_launch() enqueues one block_work node per grid block
 *       │
 *       ▼  (mutex-protected FIFO)
 *  ┌─────────────────────────────────────────────────────┐
 *  │              Block Work Queue                        │
 *  └──────┬──────────────┬──────────────┬───────────────┘
 *         │              │              │
 *    SM 0 ▼         SM 1 ▼         SM 2 ▼   ...  (num_sms dispatcher threads)
 *  execute_block()   execute_block()  execute_block()
 *         │              │              │
 *         └──────────────┴──────────────┘
 *                        │  enqueue thread_task per logical GPU thread
 *                        ▼  (mutex-protected FIFO)
 *  ┌────────────────────────────────────────────────────────┐
 *  │               Thread Task Queue                         │
 *  └─────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┘
 *        ▼      ▼      ▼      ▼      ▼      ▼      ▼   ...
 *   pool[0] pool[1]  ...                   pool[pool_size-1]
 *   (persistent worker threads — pool_size = num_sms × threads_per_sm)
 *
 * Pool threads are created once at vgpu_create() and reused for the lifetime
 * of the device, eliminating per-block pthread_create overhead.
 *
 * Deadlock avoidance
 * ──────────────────
 * Kernels with vgpu_syncthreads() use a pthread_barrier initialised with
 * block_size threads.  All block_size tasks must run concurrently for the
 * barrier to trigger.  This is guaranteed when pool_size >= block_size
 * (checked with an assertion in execute_block).
 */

#include "vgpu.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Thread task (one per logical GPU thread within a block) ───────────── */

typedef struct thread_task {
    vgpu_kernel_fn    fn;
    vgpu_thread_ctx   ctx;           /* fully populated before enqueue      */

    /* Per-block completion bookkeeping (shared among all tasks in a block) */
    int              *done_count;    /* decremented when kernel returns      */
    int               block_total;  /* == block_size; signals when 0        */
    pthread_mutex_t  *done_mutex;
    pthread_cond_t   *done_cond;

    struct thread_task *next;
} thread_task;

/* ─── Block work queue node ─────────────────────────────────────────────── */

typedef struct block_work {
    vgpu_kernel_fn   kernel;
    vgpu_dim3        block_idx;
    vgpu_dim3        block_dim;
    vgpu_dim3        grid_dim;
    size_t           shared_mem_bytes;
    void           **args;
    struct block_work *next;
} block_work;

/* ─── Device ────────────────────────────────────────────────────────────── */

struct vgpu_device {
    int num_sms;
    int threads_per_sm;
    int pool_size;          /* == num_sms * threads_per_sm                  */

    /* ── Block work queue (SM dispatcher level) ─────────────────────────── */
    pthread_mutex_t q_mutex;
    pthread_cond_t  q_cond;
    block_work     *q_head;
    block_work     *q_tail;
    int             q_pending;       /* blocks queued + in flight            */
    pthread_cond_t  q_done_cond;     /* broadcast when q_pending == 0        */
    int             q_shutdown;
    pthread_t      *sm_threads;      /* num_sms SM dispatcher threads        */

    /* ── Thread task queue (kernel-execution level) ──────────────────────── */
    pthread_mutex_t pool_mutex;
    pthread_cond_t  pool_avail;      /* pool workers wait here               */
    thread_task    *pool_head;
    thread_task    *pool_tail;
    int             pool_shutdown;
    pthread_t      *pool;            /* pool_size persistent worker threads  */

    /* ── Execution statistics ─────────────────────────────────────────────── */
    uint64_t blocks_executed;
    uint64_t threads_executed;
};

/* ─── Pool worker ───────────────────────────────────────────────────────── */

static void *pool_worker(void *arg)
{
    vgpu_device *dev = (vgpu_device *)arg;

    while (1) {
        pthread_mutex_lock(&dev->pool_mutex);
        while (!dev->pool_head && !dev->pool_shutdown)
            pthread_cond_wait(&dev->pool_avail, &dev->pool_mutex);

        if (dev->pool_shutdown && !dev->pool_head) {
            pthread_mutex_unlock(&dev->pool_mutex);
            break;
        }

        thread_task *t = dev->pool_head;
        dev->pool_head = t->next;
        if (!dev->pool_head) dev->pool_tail = NULL;
        pthread_mutex_unlock(&dev->pool_mutex);

        /* Execute the kernel for this logical GPU thread */
        t->fn(&t->ctx);

        /* Signal block completion */
        pthread_mutex_lock(t->done_mutex);
        int remaining = --(*t->done_count);
        if (remaining == 0)
            pthread_cond_signal(t->done_cond);
        pthread_mutex_unlock(t->done_mutex);

        free(t);
    }
    return NULL;
}

/* ─── Block executor (called by SM dispatcher) ──────────────────────────── */

static void execute_block(vgpu_device *dev, block_work *w)
{
    const int bx    = w->block_dim.x;
    const int by    = w->block_dim.y;
    const int bz    = w->block_dim.z;
    const int total = bx * by * bz;

    assert(total > 0);
    assert(total <= dev->pool_size &&
           "block_size must be <= pool_size (num_sms * threads_per_sm)");

    /* Per-block shared memory (zero-initialised) */
    void *shared_mem = NULL;
    if (w->shared_mem_bytes > 0)
        shared_mem = calloc(1, w->shared_mem_bytes);

    /* Per-block barrier for vgpu_syncthreads() */
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, (unsigned)total);

    /* Per-block completion tracking (stack-allocated; valid until wait exits) */
    int             done_count = total;
    pthread_mutex_t done_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t  done_cond  = PTHREAD_COND_INITIALIZER;

    /* Build all thread_tasks and enqueue them atomically */
    pthread_mutex_lock(&dev->pool_mutex);
    int t = 0;
    for (int z = 0; z < bz; z++)
    for (int y = 0; y < by; y++)
    for (int x = 0; x < bx; x++, t++) {
        thread_task *task = malloc(sizeof(thread_task));
        task->fn    = w->kernel;
        task->ctx   = (vgpu_thread_ctx){
            .threadIdx  = make_dim3(x, y, z),
            .blockIdx   = w->block_idx,
            .blockDim   = w->block_dim,
            .gridDim    = w->grid_dim,
            .shared_mem = shared_mem,
            .args       = w->args,
            .barrier    = &barrier,
        };
        task->done_count  = &done_count;
        task->block_total = total;
        task->done_mutex  = &done_mutex;
        task->done_cond   = &done_cond;
        task->next        = NULL;

        if (dev->pool_tail) dev->pool_tail->next = task;
        else                dev->pool_head = task;
        dev->pool_tail = task;
    }
    /* Wake enough pool threads to drain this block's tasks */
    pthread_cond_broadcast(&dev->pool_avail);
    pthread_mutex_unlock(&dev->pool_mutex);

    /* Wait for every logical thread in this block to complete */
    pthread_mutex_lock(&done_mutex);
    while (done_count > 0)
        pthread_cond_wait(&done_cond, &done_mutex);
    pthread_mutex_unlock(&done_mutex);

    /* Cleanup */
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&done_mutex);
    pthread_cond_destroy(&done_cond);
    free(shared_mem);

    /* Update device stats and notify vgpu_sync() if all blocks are done */
    pthread_mutex_lock(&dev->q_mutex);
    dev->blocks_executed++;
    dev->threads_executed += (uint64_t)total;
    dev->q_pending--;
    if (dev->q_pending == 0)
        pthread_cond_broadcast(&dev->q_done_cond);
    pthread_mutex_unlock(&dev->q_mutex);
}

/* ─── SM dispatcher thread ──────────────────────────────────────────────── */

static void *sm_worker(void *arg)
{
    vgpu_device *dev = (vgpu_device *)arg;

    while (1) {
        pthread_mutex_lock(&dev->q_mutex);
        while (!dev->q_head && !dev->q_shutdown)
            pthread_cond_wait(&dev->q_cond, &dev->q_mutex);

        if (dev->q_shutdown && !dev->q_head) {
            pthread_mutex_unlock(&dev->q_mutex);
            break;
        }

        block_work *w = dev->q_head;
        dev->q_head = w->next;
        if (!dev->q_head) dev->q_tail = NULL;
        pthread_mutex_unlock(&dev->q_mutex);

        execute_block(dev, w);
        free(w);
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

vgpu_device *vgpu_create(int num_sms, int threads_per_sm)
{
    assert(num_sms > 0 && threads_per_sm > 0);

    const int pool_size = num_sms * threads_per_sm;

    vgpu_device *dev = calloc(1, sizeof(vgpu_device));
    dev->num_sms        = num_sms;
    dev->threads_per_sm = threads_per_sm;
    dev->pool_size      = pool_size;

    pthread_mutex_init(&dev->q_mutex,     NULL);
    pthread_cond_init (&dev->q_cond,      NULL);
    pthread_cond_init (&dev->q_done_cond, NULL);
    pthread_mutex_init(&dev->pool_mutex,  NULL);
    pthread_cond_init (&dev->pool_avail,  NULL);

    /* Spawn SM dispatcher threads */
    dev->sm_threads = malloc((size_t)num_sms * sizeof(pthread_t));
    for (int i = 0; i < num_sms; i++)
        pthread_create(&dev->sm_threads[i], NULL, sm_worker, dev);

    /* Spawn persistent pool threads (created once, reused forever) */
    dev->pool = malloc((size_t)pool_size * sizeof(pthread_t));
    for (int i = 0; i < pool_size; i++)
        pthread_create(&dev->pool[i], NULL, pool_worker, dev);

    printf("[vGPU] Device created: %d SMs x %d threads/SM  "
           "(thread pool = %d)\n", num_sms, threads_per_sm, pool_size);
    return dev;
}

void vgpu_destroy(vgpu_device *dev)
{
    vgpu_sync(dev);

    /* Shutdown SM dispatchers */
    pthread_mutex_lock(&dev->q_mutex);
    dev->q_shutdown = 1;
    pthread_cond_broadcast(&dev->q_cond);
    pthread_mutex_unlock(&dev->q_mutex);
    for (int i = 0; i < dev->num_sms; i++)
        pthread_join(dev->sm_threads[i], NULL);

    /* Shutdown pool workers */
    pthread_mutex_lock(&dev->pool_mutex);
    dev->pool_shutdown = 1;
    pthread_cond_broadcast(&dev->pool_avail);
    pthread_mutex_unlock(&dev->pool_mutex);
    for (int i = 0; i < dev->pool_size; i++)
        pthread_join(dev->pool[i], NULL);

    pthread_mutex_destroy(&dev->q_mutex);
    pthread_cond_destroy (&dev->q_cond);
    pthread_cond_destroy (&dev->q_done_cond);
    pthread_mutex_destroy(&dev->pool_mutex);
    pthread_cond_destroy (&dev->pool_avail);
    free(dev->sm_threads);
    free(dev->pool);
    free(dev);
}

void *vgpu_malloc(size_t size)
{
    void *p = malloc(size);
    assert(p != NULL);
    return p;
}

void  vgpu_free(void *ptr)                                   { free(ptr); }
void  vgpu_memcpy_h2d(void *dst, const void *src, size_t n) { memcpy(dst, src, n); }
void  vgpu_memcpy_d2h(void *dst, const void *src, size_t n) { memcpy(dst, src, n); }

void vgpu_launch(vgpu_device   *dev,
                 vgpu_kernel_fn kernel,
                 vgpu_dim3      grid_dim,
                 vgpu_dim3      block_dim,
                 size_t         shared_mem_bytes,
                 void         **args)
{
    const int gx           = grid_dim.x, gy = grid_dim.y, gz = grid_dim.z;
    const int total_blocks = gx * gy * gz;
    assert(total_blocks > 0);

    /* Reserve pending count atomically before enqueuing — prevents a race
     * where SM workers drain the queue and signal "done" before all nodes
     * have been inserted. */
    pthread_mutex_lock(&dev->q_mutex);
    dev->q_pending += total_blocks;
    pthread_mutex_unlock(&dev->q_mutex);

    for (int z = 0; z < gz; z++)
    for (int y = 0; y < gy; y++)
    for (int x = 0; x < gx; x++) {
        block_work *w = malloc(sizeof(block_work));
        w->kernel           = kernel;
        w->block_idx        = make_dim3(x, y, z);
        w->block_dim        = block_dim;
        w->grid_dim         = grid_dim;
        w->shared_mem_bytes = shared_mem_bytes;
        w->args             = args;
        w->next             = NULL;

        pthread_mutex_lock(&dev->q_mutex);
        if (dev->q_tail) dev->q_tail->next = w;
        else             dev->q_head = w;
        dev->q_tail = w;
        pthread_cond_signal(&dev->q_cond);
        pthread_mutex_unlock(&dev->q_mutex);
    }
}

void vgpu_sync(vgpu_device *dev)
{
    pthread_mutex_lock(&dev->q_mutex);
    while (dev->q_pending > 0)
        pthread_cond_wait(&dev->q_done_cond, &dev->q_mutex);
    pthread_mutex_unlock(&dev->q_mutex);
}

void vgpu_print_stats(vgpu_device *dev)
{
    printf("\n[vGPU] -- Execution Statistics ----------------------------------\n");
    printf("  Streaming Multiprocessors : %d\n",   dev->num_sms);
    printf("  Threads per SM            : %d\n",   dev->threads_per_sm);
    printf("  Thread pool size          : %d\n",   dev->pool_size);
    printf("  Blocks  executed          : %llu\n", (unsigned long long)dev->blocks_executed);
    printf("  Threads executed          : %llu\n", (unsigned long long)dev->threads_executed);
    printf("[vGPU] ----------------------------------------------------------\n\n");
}
