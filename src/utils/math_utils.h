/**
 * Consciousness Emulator - Math Utilities
 * 
 * High-performance mathematical functions for vector operations,
 * similarity computations, and neural network operations.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#ifndef CE_MATH_UTILS_H
#define CE_MATH_UTILS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

/**
 * Compute dot product of two vectors
 * @param a First vector
 * @param b Second vector
 * @param n Vector length
 * @return Dot product
 */
float ce_dot_product(const float *a, const float *b, size_t n);

/**
 * Compute L2 norm of a vector
 * @param a Vector
 * @param n Vector length
 * @return L2 norm
 */
float ce_l2_norm(const float *a, size_t n);

/**
 * Compute cosine similarity between two vectors
 * @param a First vector
 * @param b Second vector
 * @param n Vector length
 * @return Cosine similarity [-1.0, 1.0]
 */
float ce_cosine_similarity(const float *a, const float *b, size_t n);

/**
 * Compute L2 distance between two vectors
 * @param a First vector
 * @param b Second vector
 * @param n Vector length
 * @return L2 distance
 */
float ce_l2_distance(const float *a, const float *b, size_t n);

/**
 * Compute Manhattan distance between two vectors
 * @param a First vector
 * @param b Second vector
 * @param n Vector length
 * @return Manhattan distance
 */
float ce_manhattan_distance(const float *a, const float *b, size_t n);

/**
 * Normalize vector to unit length
 * @param a Vector to normalize (modified in place)
 * @param n Vector length
 * @return CE_SUCCESS on success, error code otherwise
 */
int ce_normalize_vector(float *a, size_t n);

/**
 * Add two vectors: c = a + b
 * @param a First vector
 * @param b Second vector
 * @param c Output vector
 * @param n Vector length
 */
void ce_vector_add(const float *a, const float *b, float *c, size_t n);

/**
 * Subtract two vectors: c = a - b
 * @param a First vector
 * @param b Second vector
 * @param c Output vector
 * @param n Vector length
 */
void ce_vector_subtract(const float *a, const float *b, float *c, size_t n);

/**
 * Scale vector: b = a * scale
 * @param a Input vector
 * @param scale Scaling factor
 * @param b Output vector
 * @param n Vector length
 */
void ce_vector_scale(const float *a, float scale, float *b, size_t n);

/**
 * Compute weighted average of vectors
 * @param vectors Array of vectors
 * @param weights Array of weights
 * @param result Output vector
 * @param n_vectors Number of vectors
 * @param n_dim Vector dimension
 */
void ce_weighted_average(const float **vectors, const float *weights,
                        float *result, size_t n_vectors, size_t n_dim);

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

/**
 * Matrix-vector multiplication: y = A * x
 * @param A Matrix (row-major order)
 * @param x Input vector
 * @param y Output vector
 * @param m Number of rows
 * @param n Number of columns
 */
void ce_matrix_vector_multiply(const float *A, const float *x, float *y,
                              size_t m, size_t n);

/**
 * Matrix multiplication: C = A * B
 * @param A First matrix
 * @param B Second matrix
 * @param C Output matrix
 * @param m Rows of A
 * @param k Columns of A / Rows of B
 * @param n Columns of B
 */
void ce_matrix_multiply(const float *A, const float *B, float *C,
                       size_t m, size_t k, size_t n);

/**
 * Transpose matrix
 * @param A Input matrix
 * @param A_T Output transposed matrix
 * @param m Number of rows
 * @param n Number of columns
 */
void ce_matrix_transpose(const float *A, float *A_T, size_t m, size_t n);

/* ============================================================================
 * Activation Functions
 * ============================================================================ */

/**
 * Sigmoid activation function
 * @param x Input value
 * @return Sigmoid(x)
 */
float ce_sigmoid(float x);

/**
 * Tanh activation function
 * @param x Input value
 * @return Tanh(x)
 */
float ce_tanh(float x);

/**
 * ReLU activation function
 * @param x Input value
 * @return ReLU(x)
 */
float ce_relu(float x);

/**
 * Leaky ReLU activation function
 * @param x Input value
 * @param alpha Leak parameter
 * @return LeakyReLU(x)
 */
float ce_leaky_relu(float x, float alpha);

/**
 * Softmax activation function
 * @param x Input vector
 * @param y Output vector
 * @param n Vector length
 */
void ce_softmax(const float *x, float *y, size_t n);

/**
 * Apply activation function to vector
 * @param x Input vector
 * @param y Output vector
 * @param n Vector length
 * @param activation Activation function type
 */
void ce_apply_activation(const float *x, float *y, size_t n, int activation);

/* ============================================================================
 * Random Number Generation
 * ============================================================================ */

/**
 * Initialize random number generator
 * @param seed Random seed
 */
void ce_random_init(uint32_t seed);

/**
 * Generate random float in range [0, 1)
 * @return Random float
 */
float ce_random_float(void);

/**
 * Generate random float in range [min, max)
 * @param min Minimum value
 * @param max Maximum value
 * @return Random float
 */
float ce_random_float_range(float min, float max);

/**
 * Generate random normal (Gaussian) number
 * @param mean Mean value
 * @param std Standard deviation
 * @return Random normal number
 */
float ce_random_normal(float mean, float std);

/**
 * Generate random vector with normal distribution
 * @param vector Output vector
 * @param n Vector length
 * @param mean Mean value
 * @param std Standard deviation
 */
void ce_random_normal_vector(float *vector, size_t n, float mean, float std);

/**
 * Generate random vector with uniform distribution
 * @param vector Output vector
 * @param n Vector length
 * @param min Minimum value
 * @param max Maximum value
 */
void ce_random_uniform_vector(float *vector, size_t n, float min, float max);

/* ============================================================================
 * Statistical Functions
 * ============================================================================ */

/**
 * Compute mean of vector
 * @param x Input vector
 * @param n Vector length
 * @return Mean value
 */
float ce_mean(const float *x, size_t n);

/**
 * Compute variance of vector
 * @param x Input vector
 * @param n Vector length
 * @return Variance
 */
float ce_variance(const float *x, size_t n);

/**
 * Compute standard deviation of vector
 * @param x Input vector
 * @param n Vector length
 * @return Standard deviation
 */
float ce_std_dev(const float *x, size_t n);

/**
 * Compute correlation coefficient between two vectors
 * @param x First vector
 * @param y Second vector
 * @param n Vector length
 * @return Correlation coefficient [-1.0, 1.0]
 */
float ce_correlation(const float *x, const float *y, size_t n);

/* ============================================================================
 * Optimization Functions
 * ============================================================================ */

/**
 * Find index of maximum element in vector
 * @param x Input vector
 * @param n Vector length
 * @return Index of maximum element
 */
size_t ce_argmax(const float *x, size_t n);

/**
 * Find index of minimum element in vector
 * @param x Input vector
 * @param n Vector length
 * @return Index of minimum element
 */
size_t ce_argmin(const float *x, size_t n);

/**
 * Find maximum element in vector
 * @param x Input vector
 * @param n Vector length
 * @return Maximum element
 */
float ce_max(const float *x, size_t n);

/**
 * Find minimum element in vector
 * @param x Input vector
 * @param n Vector length
 * @return Minimum element
 */
float ce_min(const float *x, size_t n);

/**
 * Sort vector in ascending order (in place)
 * @param x Vector to sort
 * @param n Vector length
 */
void ce_sort_ascending(float *x, size_t n);

/**
 * Sort vector in descending order (in place)
 * @param x Vector to sort
 * @param n Vector length
 */
void ce_sort_descending(float *x, size_t n);

/* ============================================================================
 * Distance Metrics
 * ============================================================================ */

/**
 * Compute Hamming distance between two binary vectors
 * @param a First vector
 * @param b Second vector
 * @param n Vector length
 * @return Hamming distance
 */
size_t ce_hamming_distance(const uint8_t *a, const uint8_t *b, size_t n);

/**
 * Compute Jaccard similarity between two sets
 * @param a First set (binary vector)
 * @param b Second set (binary vector)
 * @param n Vector length
 * @return Jaccard similarity [0.0, 1.0]
 */
float ce_jaccard_similarity(const uint8_t *a, const uint8_t *b, size_t n);

/**
 * Compute KL divergence between two probability distributions
 * @param p First distribution
 * @param q Second distribution
 * @param n Vector length
 * @return KL divergence
 */
float ce_kl_divergence(const float *p, const float *q, size_t n);

/* ============================================================================
 * Neural Network Utilities
 * ============================================================================ */

/**
 * Compute cross-entropy loss
 * @param predictions Predicted probabilities
 * @param targets Target probabilities
 * @param n Vector length
 * @return Cross-entropy loss
 */
float ce_cross_entropy_loss(const float *predictions, const float *targets, size_t n);

/**
 * Compute mean squared error
 * @param predictions Predicted values
 * @param targets Target values
 * @param n Vector length
 * @return Mean squared error
 */
float ce_mse_loss(const float *predictions, const float *targets, size_t n);

/**
 * Apply dropout to vector
 * @param x Input vector
 * @param y Output vector
 * @param n Vector length
 * @param dropout_rate Dropout rate [0.0, 1.0]
 * @param training Whether in training mode
 */
void ce_apply_dropout(const float *x, float *y, size_t n, float dropout_rate, bool training);

/**
 * Apply batch normalization
 * @param x Input vector
 * @param y Output vector
 * @param n Vector length
 * @param gamma Scale parameter
 * @param beta Shift parameter
 * @param mean Batch mean
 * @param var Batch variance
 * @param epsilon Small constant for numerical stability
 */
void ce_batch_normalize(const float *x, float *y, size_t n, float gamma, float beta,
                       float mean, float var, float epsilon);

/* ============================================================================
 * SIMD Optimizations
 * ============================================================================ */

/**
 * Check if SIMD instructions are available
 * @return True if SIMD is available
 */
bool ce_simd_available(void);

/**
 * Get optimal vector size for SIMD operations
 * @return Optimal vector size
 */
size_t ce_simd_vector_size(void);

/**
 * SIMD-optimized dot product
 * @param a First vector
 * @param b Second vector
 * @param n Vector length
 * @return Dot product
 */
float ce_simd_dot_product(const float *a, const float *b, size_t n);

/**
 * SIMD-optimized vector addition
 * @param a First vector
 * @param b Second vector
 * @param c Output vector
 * @param n Vector length
 */
void ce_simd_vector_add(const float *a, const float *b, float *c, size_t n);

/**
 * SIMD-optimized matrix-vector multiplication
 * @param A Matrix
 * @param x Input vector
 * @param y Output vector
 * @param m Number of rows
 * @param n Number of columns
 */
void ce_simd_matrix_vector_multiply(const float *A, const float *x, float *y,
                                   size_t m, size_t n);

#endif /* CE_MATH_UTILS_H */
