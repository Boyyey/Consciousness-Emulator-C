/**
 * Consciousness Emulator v1.1 - CUDA Acceleration
 * 
 * CUDA kernels for high-performance GPU acceleration of cognitive operations.
 * Provides massive parallelization for vector operations and neural computations.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#ifndef CE_CUDA_KERNELS_H
#define CE_CUDA_KERNELS_H

#include "../../include/consciousness.h"
#include <stdbool.h>

/* ============================================================================
 * CUDA Configuration
 * ============================================================================ */

#define CE_CUDA_MAX_DEVICES 8
#define CE_CUDA_MAX_THREADS_PER_BLOCK 1024
#define CE_CUDA_DEFAULT_BLOCK_SIZE 256
#define CE_CUDA_MAX_SHARED_MEMORY 49152

/* ============================================================================
 * CUDA Device Management
 * ============================================================================ */

typedef struct {
    int device_id;
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int max_threads_per_block;
    int max_threads_per_dim[3];
    int max_grid_size[3];
    size_t shared_memory_per_block;
    int multiprocessor_count;
    int clock_rate;
    bool is_available;
} ce_cuda_device_t;

typedef struct {
    ce_cuda_device_t devices[CE_CUDA_MAX_DEVICES];
    int device_count;
    int current_device;
    bool is_initialized;
} ce_cuda_context_t;

/* ============================================================================
 * CUDA Memory Management
 * ============================================================================ */

typedef struct {
    void *d_ptr;           /* Device pointer */
    size_t size;           /* Allocated size in bytes */
    bool is_allocated;     /* Whether memory is allocated */
} ce_cuda_memory_t;

typedef struct {
    ce_cuda_memory_t *allocations;
    size_t count;
    size_t capacity;
    size_t total_allocated;
} ce_cuda_memory_pool_t;

/* ============================================================================
 * CUDA Stream Management
 * ============================================================================ */

typedef struct {
    void *stream;          /* CUDA stream */
    bool is_active;        /* Whether stream is active */
    int priority;          /* Stream priority */
} ce_cuda_stream_t;

typedef struct {
    ce_cuda_stream_t *streams;
    size_t count;
    size_t capacity;
} ce_cuda_stream_pool_t;

/* ============================================================================
 * CUDA API
 * ============================================================================ */

/**
 * Initialize CUDA context
 * @return CUDA context or NULL on error
 */
ce_cuda_context_t *ce_cuda_init(void);

/**
 * Free CUDA context
 * @param context CUDA context
 */
void ce_cuda_free(ce_cuda_context_t *context);

/**
 * Get CUDA device information
 * @param context CUDA context
 * @param device_id Device ID
 * @return Device information or NULL if not found
 */
const ce_cuda_device_t *ce_cuda_get_device_info(ce_cuda_context_t *context, int device_id);

/**
 * Set current CUDA device
 * @param context CUDA context
 * @param device_id Device ID
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_set_device(ce_cuda_context_t *context, int device_id);

/**
 * Check if CUDA is available
 * @return True if CUDA is available
 */
bool ce_cuda_is_available(void);

/* ============================================================================
 * CUDA Memory Management API
 * ============================================================================ */

/**
 * Create CUDA memory pool
 * @param initial_capacity Initial capacity
 * @return Memory pool or NULL on error
 */
ce_cuda_memory_pool_t *ce_cuda_memory_pool_create(size_t initial_capacity);

/**
 * Free CUDA memory pool
 * @param pool Memory pool
 */
void ce_cuda_memory_pool_free(ce_cuda_memory_pool_t *pool);

/**
 * Allocate CUDA memory
 * @param pool Memory pool
 * @param size Size in bytes
 * @return Memory handle or NULL on error
 */
ce_cuda_memory_t *ce_cuda_allocate(ce_cuda_memory_pool_t *pool, size_t size);

/**
 * Free CUDA memory
 * @param pool Memory pool
 * @param memory Memory handle
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_free_memory(ce_cuda_memory_pool_t *pool, ce_cuda_memory_t *memory);

/**
 * Copy data from host to device
 * @param dst Device pointer
 * @param src Host pointer
 * @param size Size in bytes
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_memcpy_h2d(void *dst, const void *src, size_t size);

/**
 * Copy data from device to host
 * @param dst Host pointer
 * @param src Device pointer
 * @param size Size in bytes
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_memcpy_d2h(void *dst, const void *src, size_t size);

/**
 * Copy data from device to device
 * @param dst Destination device pointer
 * @param src Source device pointer
 * @param size Size in bytes
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_memcpy_d2d(void *dst, const void *src, size_t size);

/* ============================================================================
 * CUDA Stream Management API
 * ============================================================================ */

/**
 * Create CUDA stream pool
 * @param num_streams Number of streams
 * @return Stream pool or NULL on error
 */
ce_cuda_stream_pool_t *ce_cuda_stream_pool_create(size_t num_streams);

/**
 * Free CUDA stream pool
 * @param pool Stream pool
 */
void ce_cuda_stream_pool_free(ce_cuda_stream_pool_t *pool);

/**
 * Get available stream
 * @param pool Stream pool
 * @return Stream or NULL if none available
 */
ce_cuda_stream_t *ce_cuda_get_stream(ce_cuda_stream_pool_t *pool);

/**
 * Synchronize stream
 * @param stream Stream to synchronize
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_stream_synchronize(ce_cuda_stream_t *stream);

/**
 * Synchronize all streams
 * @param pool Stream pool
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_synchronize_all(ce_cuda_stream_pool_t *pool);

/* ============================================================================
 * CUDA Kernel Launchers
 * ============================================================================ */

/**
 * Launch vector addition kernel
 * @param a First vector (device)
 * @param b Second vector (device)
 * @param c Result vector (device)
 * @param n Vector length
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_vector_add(const float *a, const float *b, float *c, 
                              size_t n, ce_cuda_stream_t *stream);

/**
 * Launch vector subtraction kernel
 * @param a First vector (device)
 * @param b Second vector (device)
 * @param c Result vector (device)
 * @param n Vector length
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_vector_subtract(const float *a, const float *b, float *c, 
                                   size_t n, ce_cuda_stream_t *stream);

/**
 * Launch vector scaling kernel
 * @param a Input vector (device)
 * @param scale Scaling factor
 * @param b Result vector (device)
 * @param n Vector length
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_vector_scale(const float *a, float scale, float *b, 
                                size_t n, ce_cuda_stream_t *stream);

/**
 * Launch dot product kernel
 * @param a First vector (device)
 * @param b Second vector (device)
 * @param result Result (host)
 * @param n Vector length
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_dot_product(const float *a, const float *b, float *result, 
                               size_t n, ce_cuda_stream_t *stream);

/**
 * Launch L2 norm kernel
 * @param a Input vector (device)
 * @param result Result (host)
 * @param n Vector length
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_l2_norm(const float *a, float *result, size_t n, 
                           ce_cuda_stream_t *stream);

/**
 * Launch cosine similarity kernel
 * @param a First vector (device)
 * @param b Second vector (device)
 * @param result Result (host)
 * @param n Vector length
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_cosine_similarity(const float *a, const float *b, float *result, 
                                     size_t n, ce_cuda_stream_t *stream);

/**
 * Launch matrix-vector multiplication kernel
 * @param A Matrix (device)
 * @param x Vector (device)
 * @param y Result vector (device)
 * @param m Matrix rows
 * @param n Matrix columns
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_matrix_vector_multiply(const float *A, const float *x, float *y,
                                          size_t m, size_t n, ce_cuda_stream_t *stream);

/**
 * Launch matrix multiplication kernel
 * @param A First matrix (device)
 * @param B Second matrix (device)
 * @param C Result matrix (device)
 * @param m Rows of A
 * @param k Columns of A / Rows of B
 * @param n Columns of B
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_matrix_multiply(const float *A, const float *B, float *C,
                                   size_t m, size_t k, size_t n, ce_cuda_stream_t *stream);

/**
 * Launch softmax kernel
 * @param x Input vector (device)
 * @param y Output vector (device)
 * @param n Vector length
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_softmax(const float *x, float *y, size_t n, 
                           ce_cuda_stream_t *stream);

/**
 * Launch ReLU kernel
 * @param x Input vector (device)
 * @param y Output vector (device)
 * @param n Vector length
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_relu(const float *x, float *y, size_t n, 
                        ce_cuda_stream_t *stream);

/**
 * Launch sigmoid kernel
 * @param x Input vector (device)
 * @param y Output vector (device)
 * @param n Vector length
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_sigmoid(const float *x, float *y, size_t n, 
                           ce_cuda_stream_t *stream);

/* ============================================================================
 * CUDA Neural Network Kernels
 * ============================================================================ */

/**
 * Launch neural network forward pass kernel
 * @param input Input tensor (device)
 * @param weights Weight tensors (device)
 * @param biases Bias tensors (device)
 * @param output Output tensor (device)
 * @param layer_sizes Layer sizes array
 * @param num_layers Number of layers
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_neural_forward(const float *input, const float **weights,
                                  const float **biases, float *output,
                                  const size_t *layer_sizes, int num_layers,
                                  ce_cuda_stream_t *stream);

/**
 * Launch attention mechanism kernel
 * @param queries Query tensor (device)
 * @param keys Key tensor (device)
 * @param values Value tensor (device)
 * @param output Attention output (device)
 * @param seq_len Sequence length
 * @param d_model Model dimension
 * @param num_heads Number of attention heads
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_attention(const float *queries, const float *keys,
                             const float *values, float *output,
                             size_t seq_len, size_t d_model, size_t num_heads,
                             ce_cuda_stream_t *stream);

/**
 * Launch transformer block kernel
 * @param input Input tensor (device)
 * @param output Output tensor (device)
 * @param weights Weight tensors (device)
 * @param seq_len Sequence length
 * @param d_model Model dimension
 * @param num_heads Number of attention heads
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_transformer_block(const float *input, float *output,
                                     const float **weights, size_t seq_len,
                                     size_t d_model, size_t num_heads,
                                     ce_cuda_stream_t *stream);

/* ============================================================================
 * CUDA Cognitive Processing Kernels
 * ============================================================================ */

/**
 * Launch saliency computation kernel
 * @param items Item embeddings (device)
 * @param goals Goal embeddings (device)
 * @param saliencies Output saliencies (device)
 * @param num_items Number of items
 * @param embedding_dim Embedding dimension
 * @param num_goals Number of goals
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_compute_saliency(const float *items, const float *goals,
                                    float *saliencies, size_t num_items,
                                    size_t embedding_dim, size_t num_goals,
                                    ce_cuda_stream_t *stream);

/**
 * Launch attention arbitration kernel
 * @param saliencies Input saliencies (device)
 * @param selected_indices Output selected indices (device)
 * @param num_items Number of items
 * @param max_selected Maximum number to select
 * @param threshold Selection threshold
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_arbitrate_attention(const float *saliencies,
                                       int *selected_indices, size_t num_items,
                                       size_t max_selected, float threshold,
                                       ce_cuda_stream_t *stream);

/**
 * Launch memory consolidation kernel
 * @param episodes Episode embeddings (device)
 * @param similarities Output similarities (device)
 * @param num_episodes Number of episodes
 * @param embedding_dim Embedding dimension
 * @param stream CUDA stream
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_consolidate_memories(const float *episodes, float *similarities,
                                        size_t num_episodes, size_t embedding_dim,
                                        ce_cuda_stream_t *stream);

/* ============================================================================
 * CUDA Utility Functions
 * ============================================================================ */

/**
 * Get optimal block size for kernel
 * @param n Problem size
 * @param max_threads Maximum threads per block
 * @return Optimal block size
 */
int ce_cuda_get_optimal_block_size(size_t n, int max_threads);

/**
 * Get optimal grid size for kernel
 * @param n Problem size
 * @param block_size Block size
 * @return Optimal grid size
 */
int ce_cuda_get_optimal_grid_size(size_t n, int block_size);

/**
 * Benchmark CUDA kernel performance
 * @param kernel_func Kernel function to benchmark
 * @param args Kernel arguments
 * @param num_iterations Number of benchmark iterations
 * @param stats Output benchmark statistics
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_benchmark_kernel(void (*kernel_func)(void*), void *args,
                                    size_t num_iterations,
                                    struct {
                                        double avg_time;
                                        double min_time;
                                        double max_time;
                                        double throughput;
                                    } *stats);

/**
 * Get CUDA memory usage statistics
 * @param context CUDA context
 * @param stats Output memory statistics
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_get_memory_stats(ce_cuda_context_t *context, struct {
    size_t total_memory;
    size_t free_memory;
    size_t used_memory;
    size_t allocated_memory;
} *stats);

#endif /* CE_CUDA_KERNELS_H */
