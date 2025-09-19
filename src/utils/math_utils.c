/**
 * Consciousness Emulator - Math Utilities Implementation
 * 
 * High-performance mathematical functions with SIMD optimizations.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <x86intrin.h>

/* ============================================================================
 * Global State
 * ============================================================================ */

static uint32_t g_random_state = 1;

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

float ce_dot_product(const float *a, const float *b, size_t n) {
    if (!a || !b || n == 0) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    
    /* SIMD-optimized version for large vectors */
    if (n >= 8) {
        __m256 sum_vec = _mm256_setzero_ps();
        
        size_t i;
        for (i = 0; i <= n - 8; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
        }
        
        /* Horizontal sum */
        __m128 sum_128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 0),
                                   _mm256_extractf128_ps(sum_vec, 1));
        sum_128 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
        sum_128 = _mm_add_ss(sum_128, _mm_movehdup_ps(sum_128));
        sum = _mm_cvtss_f32(sum_128);
        
        /* Handle remaining elements */
        for (; i < n; i++) {
            sum += a[i] * b[i];
        }
    } else {
        /* Scalar version for small vectors */
        for (size_t i = 0; i < n; i++) {
            sum += a[i] * b[i];
        }
    }
    
    return sum;
}

float ce_l2_norm(const float *a, size_t n) {
    if (!a || n == 0) {
        return 0.0f;
    }
    
    return sqrtf(ce_dot_product(a, a, n));
}

float ce_cosine_similarity(const float *a, const float *b, size_t n) {
    if (!a || !b || n == 0) {
        return 0.0f;
    }
    
    float dot_product = ce_dot_product(a, b, n);
    float norm_a = ce_l2_norm(a, n);
    float norm_b = ce_l2_norm(b, n);
    
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }
    
    return dot_product / (norm_a * norm_b);
}

float ce_l2_distance(const float *a, const float *b, size_t n) {
    if (!a || !b || n == 0) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    
    /* SIMD-optimized version */
    if (n >= 8) {
        __m256 sum_vec = _mm256_setzero_ps();
        
        size_t i;
        for (i = 0; i <= n - 8; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
        }
        
        /* Horizontal sum */
        __m128 sum_128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 0),
                                   _mm256_extractf128_ps(sum_vec, 1));
        sum_128 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
        sum_128 = _mm_add_ss(sum_128, _mm_movehdup_ps(sum_128));
        sum = _mm_cvtss_f32(sum_128);
        
        /* Handle remaining elements */
        for (; i < n; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
    } else {
        /* Scalar version */
        for (size_t i = 0; i < n; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
    }
    
    return sqrtf(sum);
}

float ce_manhattan_distance(const float *a, const float *b, size_t n) {
    if (!a || !b || n == 0) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    
    /* SIMD-optimized version */
    if (n >= 8) {
        __m256 sum_vec = _mm256_setzero_ps();
        
        size_t i;
        for (i = 0; i <= n - 8; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 diff = _mm256_sub_ps(va, vb);
            __m256 abs_diff = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff);
            sum_vec = _mm256_add_ps(sum_vec, abs_diff);
        }
        
        /* Horizontal sum */
        __m128 sum_128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 0),
                                   _mm256_extractf128_ps(sum_vec, 1));
        sum_128 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
        sum_128 = _mm_add_ss(sum_128, _mm_movehdup_ps(sum_128));
        sum = _mm_cvtss_f32(sum_128);
        
        /* Handle remaining elements */
        for (; i < n; i++) {
            sum += fabsf(a[i] - b[i]);
        }
    } else {
        /* Scalar version */
        for (size_t i = 0; i < n; i++) {
            sum += fabsf(a[i] - b[i]);
        }
    }
    
    return sum;
}

int ce_normalize_vector(float *a, size_t n) {
    if (!a || n == 0) {
        return -1;
    }
    
    float norm = ce_l2_norm(a, n);
    if (norm == 0.0f) {
        return -1; /* Cannot normalize zero vector */
    }
    
    ce_vector_scale(a, 1.0f / norm, a, n);
    return 0;
}

void ce_vector_add(const float *a, const float *b, float *c, size_t n) {
    if (!a || !b || !c || n == 0) {
        return;
    }
    
    /* SIMD-optimized version */
    if (n >= 8) {
        size_t i;
        for (i = 0; i <= n - 8; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&c[i], vc);
        }
        
        /* Handle remaining elements */
        for (; i < n; i++) {
            c[i] = a[i] + b[i];
        }
    } else {
        /* Scalar version */
        for (size_t i = 0; i < n; i++) {
            c[i] = a[i] + b[i];
        }
    }
}

void ce_vector_subtract(const float *a, const float *b, float *c, size_t n) {
    if (!a || !b || !c || n == 0) {
        return;
    }
    
    /* SIMD-optimized version */
    if (n >= 8) {
        size_t i;
        for (i = 0; i <= n - 8; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vc = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(&c[i], vc);
        }
        
        /* Handle remaining elements */
        for (; i < n; i++) {
            c[i] = a[i] - b[i];
        }
    } else {
        /* Scalar version */
        for (size_t i = 0; i < n; i++) {
            c[i] = a[i] - b[i];
        }
    }
}

void ce_vector_scale(const float *a, float scale, float *b, size_t n) {
    if (!a || !b || n == 0) {
        return;
    }
    
    /* SIMD-optimized version */
    if (n >= 8) {
        __m256 scale_vec = _mm256_set1_ps(scale);
        
        size_t i;
        for (i = 0; i <= n - 8; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_mul_ps(va, scale_vec);
            _mm256_storeu_ps(&b[i], vb);
        }
        
        /* Handle remaining elements */
        for (; i < n; i++) {
            b[i] = a[i] * scale;
        }
    } else {
        /* Scalar version */
        for (size_t i = 0; i < n; i++) {
            b[i] = a[i] * scale;
        }
    }
}

void ce_weighted_average(const float **vectors, const float *weights,
                        float *result, size_t n_vectors, size_t n_dim) {
    if (!vectors || !weights || !result || n_vectors == 0 || n_dim == 0) {
        return;
    }
    
    /* Initialize result to zero */
    memset(result, 0, n_dim * sizeof(float));
    
    /* Compute weighted sum */
    for (size_t i = 0; i < n_vectors; i++) {
        if (vectors[i]) {
            for (size_t j = 0; j < n_dim; j++) {
                result[j] += vectors[i][j] * weights[i];
            }
        }
    }
}

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

void ce_matrix_vector_multiply(const float *A, const float *x, float *y,
                              size_t m, size_t n) {
    if (!A || !x || !y || m == 0 || n == 0) {
        return;
    }
    
    /* SIMD-optimized version */
    if (n >= 8) {
        for (size_t i = 0; i < m; i++) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            size_t j;
            for (j = 0; j <= n - 8; j += 8) {
                __m256 va = _mm256_loadu_ps(&A[i * n + j]);
                __m256 vx = _mm256_loadu_ps(&x[j]);
                sum_vec = _mm256_fmadd_ps(va, vx, sum_vec);
            }
            
            /* Horizontal sum */
            __m128 sum_128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 0),
                                       _mm256_extractf128_ps(sum_vec, 1));
            sum_128 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
            sum_128 = _mm_add_ss(sum_128, _mm_movehdup_ps(sum_128));
            y[i] = _mm_cvtss_f32(sum_128);
            
            /* Handle remaining elements */
            for (; j < n; j++) {
                y[i] += A[i * n + j] * x[j];
            }
        }
    } else {
        /* Scalar version */
        for (size_t i = 0; i < m; i++) {
            y[i] = 0.0f;
            for (size_t j = 0; j < n; j++) {
                y[i] += A[i * n + j] * x[j];
            }
        }
    }
}

void ce_matrix_multiply(const float *A, const float *B, float *C,
                       size_t m, size_t k, size_t n) {
    if (!A || !B || !C || m == 0 || k == 0 || n == 0) {
        return;
    }
    
    /* Initialize result matrix */
    memset(C, 0, m * n * sizeof(float));
    
    /* SIMD-optimized version */
    if (k >= 8) {
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                __m256 sum_vec = _mm256_setzero_ps();
                
                size_t l;
                for (l = 0; l <= k - 8; l += 8) {
                    __m256 va = _mm256_loadu_ps(&A[i * k + l]);
                    __m256 vb = _mm256_loadu_ps(&B[l * n + j]);
                    sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
                }
                
                /* Horizontal sum */
                __m128 sum_128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 0),
                                           _mm256_extractf128_ps(sum_vec, 1));
                sum_128 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
                sum_128 = _mm_add_ss(sum_128, _mm_movehdup_ps(sum_128));
                C[i * n + j] = _mm_cvtss_f32(sum_128);
                
                /* Handle remaining elements */
                for (; l < k; l++) {
                    C[i * n + j] += A[i * k + l] * B[l * n + j];
                }
            }
        }
    } else {
        /* Scalar version */
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                C[i * n + j] = 0.0f;
                for (size_t l = 0; l < k; l++) {
                    C[i * n + j] += A[i * k + l] * B[l * n + j];
                }
            }
        }
    }
}

void ce_matrix_transpose(const float *A, float *A_T, size_t m, size_t n) {
    if (!A || !A_T || m == 0 || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            A_T[j * m + i] = A[i * n + j];
        }
    }
}

/* ============================================================================
 * Activation Functions
 * ============================================================================ */

float ce_sigmoid(float x) {
    /* Clamp x to prevent overflow */
    if (x > 88.0f) return 1.0f;
    if (x < -88.0f) return 0.0f;
    
    return 1.0f / (1.0f + expf(-x));
}

float ce_tanh(float x) {
    return tanhf(x);
}

float ce_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float ce_leaky_relu(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

void ce_softmax(const float *x, float *y, size_t n) {
    if (!x || !y || n == 0) {
        return;
    }
    
    /* Find maximum for numerical stability */
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    /* Compute exponentials and sum */
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        y[i] = expf(x[i] - max_val);
        sum += y[i];
    }
    
    /* Normalize */
    if (sum > 0.0f) {
        for (size_t i = 0; i < n; i++) {
            y[i] /= sum;
        }
    }
}

void ce_apply_activation(const float *x, float *y, size_t n, int activation) {
    if (!x || !y || n == 0) {
        return;
    }
    
    switch (activation) {
        case 0: /* Sigmoid */
            for (size_t i = 0; i < n; i++) {
                y[i] = ce_sigmoid(x[i]);
            }
            break;
            
        case 1: /* Tanh */
            for (size_t i = 0; i < n; i++) {
                y[i] = ce_tanh(x[i]);
            }
            break;
            
        case 2: /* ReLU */
            for (size_t i = 0; i < n; i++) {
                y[i] = ce_relu(x[i]);
            }
            break;
            
        case 3: /* Leaky ReLU */
            for (size_t i = 0; i < n; i++) {
                y[i] = ce_leaky_relu(x[i], 0.01f);
            }
            break;
            
        default:
            /* Identity */
            memcpy(y, x, n * sizeof(float));
            break;
    }
}

/* ============================================================================
 * Random Number Generation
 * ============================================================================ */

void ce_random_init(uint32_t seed) {
    g_random_state = seed;
}

float ce_random_float(void) {
    /* Linear congruential generator */
    g_random_state = g_random_state * 1103515245 + 12345;
    return (float)(g_random_state & 0x7FFFFFFF) / 2147483647.0f;
}

float ce_random_float_range(float min, float max) {
    return min + ce_random_float() * (max - min);
}

float ce_random_normal(float mean, float std) {
    /* Box-Muller transform */
    static bool has_spare = false;
    static float spare;
    
    if (has_spare) {
        has_spare = false;
        return mean + std * spare;
    }
    
    has_spare = true;
    float u = ce_random_float();
    float v = ce_random_float();
    
    float mag = std * sqrtf(-2.0f * logf(u));
    spare = mag * cosf(2.0f * M_PI * v);
    return mean + mag * sinf(2.0f * M_PI * v);
}

void ce_random_normal_vector(float *vector, size_t n, float mean, float std) {
    if (!vector || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        vector[i] = ce_random_normal(mean, std);
    }
}

void ce_random_uniform_vector(float *vector, size_t n, float min, float max) {
    if (!vector || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        vector[i] = ce_random_float_range(min, max);
    }
}

/* ============================================================================
 * Statistical Functions
 * ============================================================================ */

float ce_mean(const float *x, size_t n) {
    if (!x || n == 0) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += x[i];
    }
    
    return sum / n;
}

float ce_variance(const float *x, size_t n) {
    if (!x || n == 0) {
        return 0.0f;
    }
    
    float mean_val = ce_mean(x, n);
    float sum_sq_diff = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float diff = x[i] - mean_val;
        sum_sq_diff += diff * diff;
    }
    
    return sum_sq_diff / n;
}

float ce_std_dev(const float *x, size_t n) {
    return sqrtf(ce_variance(x, n));
}

float ce_correlation(const float *x, const float *y, size_t n) {
    if (!x || !y || n == 0) {
        return 0.0f;
    }
    
    float mean_x = ce_mean(x, n);
    float mean_y = ce_mean(y, n);
    
    float numerator = 0.0f;
    float sum_sq_x = 0.0f;
    float sum_sq_y = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float diff_x = x[i] - mean_x;
        float diff_y = y[i] - mean_y;
        
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }
    
    float denominator = sqrtf(sum_sq_x * sum_sq_y);
    if (denominator == 0.0f) {
        return 0.0f;
    }
    
    return numerator / denominator;
}

/* ============================================================================
 * Optimization Functions
 * ============================================================================ */

size_t ce_argmax(const float *x, size_t n) {
    if (!x || n == 0) {
        return SIZE_MAX;
    }
    
    size_t max_idx = 0;
    float max_val = x[0];
    
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

size_t ce_argmin(const float *x, size_t n) {
    if (!x || n == 0) {
        return SIZE_MAX;
    }
    
    size_t min_idx = 0;
    float min_val = x[0];
    
    for (size_t i = 1; i < n; i++) {
        if (x[i] < min_val) {
            min_val = x[i];
            min_idx = i;
        }
    }
    
    return min_idx;
}

float ce_max(const float *x, size_t n) {
    if (!x || n == 0) {
        return 0.0f;
    }
    
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    return max_val;
}

float ce_min(const float *x, size_t n) {
    if (!x || n == 0) {
        return 0.0f;
    }
    
    float min_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] < min_val) {
            min_val = x[i];
        }
    }
    
    return min_val;
}

void ce_sort_ascending(float *x, size_t n) {
    if (!x || n == 0) {
        return;
    }
    
    /* Simple bubble sort - replace with quicksort for better performance */
    for (size_t i = 0; i < n - 1; i++) {
        for (size_t j = 0; j < n - i - 1; j++) {
            if (x[j] > x[j + 1]) {
                float temp = x[j];
                x[j] = x[j + 1];
                x[j + 1] = temp;
            }
        }
    }
}

void ce_sort_descending(float *x, size_t n) {
    if (!x || n == 0) {
        return;
    }
    
    /* Simple bubble sort - replace with quicksort for better performance */
    for (size_t i = 0; i < n - 1; i++) {
        for (size_t j = 0; j < n - i - 1; j++) {
            if (x[j] < x[j + 1]) {
                float temp = x[j];
                x[j] = x[j + 1];
                x[j + 1] = temp;
            }
        }
    }
}

/* ============================================================================
 * Distance Metrics
 * ============================================================================ */

size_t ce_hamming_distance(const uint8_t *a, const uint8_t *b, size_t n) {
    if (!a || !b || n == 0) {
        return 0;
    }
    
    size_t distance = 0;
    for (size_t i = 0; i < n; i++) {
        distance += __builtin_popcount(a[i] ^ b[i]);
    }
    
    return distance;
}

float ce_jaccard_similarity(const uint8_t *a, const uint8_t *b, size_t n) {
    if (!a || !b || n == 0) {
        return 0.0f;
    }
    
    size_t intersection = 0;
    size_t union_size = 0;
    
    for (size_t i = 0; i < n; i++) {
        uint8_t a_val = a[i];
        uint8_t b_val = b[i];
        
        intersection += __builtin_popcount(a_val & b_val);
        union_size += __builtin_popcount(a_val | b_val);
    }
    
    return union_size > 0 ? (float)intersection / union_size : 0.0f;
}

float ce_kl_divergence(const float *p, const float *q, size_t n) {
    if (!p || !q || n == 0) {
        return 0.0f;
    }
    
    float divergence = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        if (p[i] > 0.0f && q[i] > 0.0f) {
            divergence += p[i] * logf(p[i] / q[i]);
        }
    }
    
    return divergence;
}

/* ============================================================================
 * Neural Network Utilities
 * ============================================================================ */

float ce_cross_entropy_loss(const float *predictions, const float *targets, size_t n) {
    if (!predictions || !targets || n == 0) {
        return 0.0f;
    }
    
    float loss = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        if (targets[i] > 0.0f) {
            loss -= targets[i] * logf(fmaxf(predictions[i], 1e-15f));
        }
    }
    
    return loss;
}

float ce_mse_loss(const float *predictions, const float *targets, size_t n) {
    if (!predictions || !targets || n == 0) {
        return 0.0f;
    }
    
    float sum_sq_error = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float error = predictions[i] - targets[i];
        sum_sq_error += error * error;
    }
    
    return sum_sq_error / n;
}

void ce_apply_dropout(const float *x, float *y, size_t n, float dropout_rate, bool training) {
    if (!x || !y || n == 0) {
        return;
    }
    
    if (!training || dropout_rate <= 0.0f) {
        memcpy(y, x, n * sizeof(float));
        return;
    }
    
    float scale = 1.0f / (1.0f - dropout_rate);
    
    for (size_t i = 0; i < n; i++) {
        y[i] = (ce_random_float() > dropout_rate) ? x[i] * scale : 0.0f;
    }
}

void ce_batch_normalize(const float *x, float *y, size_t n, float gamma, float beta,
                       float mean, float var, float epsilon) {
    if (!x || !y || n == 0) {
        return;
    }
    
    float std = sqrtf(var + epsilon);
    
    for (size_t i = 0; i < n; i++) {
        y[i] = gamma * (x[i] - mean) / std + beta;
    }
}

/* ============================================================================
 * SIMD Optimizations
 * ============================================================================ */

bool ce_simd_available(void) {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

size_t ce_simd_vector_size(void) {
    return ce_simd_available() ? 8 : 4;
}

float ce_simd_dot_product(const float *a, const float *b, size_t n) {
    return ce_dot_product(a, b, n); /* Already SIMD-optimized */
}

void ce_simd_vector_add(const float *a, const float *b, float *c, size_t n) {
    ce_vector_add(a, b, c, n); /* Already SIMD-optimized */
}

void ce_simd_matrix_vector_multiply(const float *A, const float *x, float *y,
                                   size_t m, size_t n) {
    ce_matrix_vector_multiply(A, x, y, m, n); /* Already SIMD-optimized */
}
