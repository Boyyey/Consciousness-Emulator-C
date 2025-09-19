/**
 * Consciousness Emulator v1.1 - Neural Network Engine
 * 
 * ONNX Runtime integration for neural network inference in the CE system.
 * Provides seamless integration between symbolic reasoning and neural processing.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#ifndef CE_NEURAL_ENGINE_H
#define CE_NEURAL_ENGINE_H

#include "../../include/consciousness.h"
#include <stdbool.h>

/* ============================================================================
 * Neural Network Configuration
 * ============================================================================ */

#define CE_NEURAL_MAX_MODELS 16
#define CE_NEURAL_MAX_INPUT_SIZE 4096
#define CE_NEURAL_MAX_OUTPUT_SIZE 4096
#define CE_NEURAL_DEFAULT_BATCH_SIZE 1
#define CE_NEURAL_DEFAULT_THREADS 4

/* ============================================================================
 * Neural Network Types
 * ============================================================================ */

typedef enum {
    CE_NEURAL_MODEL_TYPE_PATTERN_RECOGNITION = 0,
    CE_NEURAL_MODEL_TYPE_ASSOCIATION,
    CE_NEURAL_MODEL_TYPE_PREDICTION,
    CE_NEURAL_MODEL_TYPE_EMBEDDING,
    CE_NEURAL_MODEL_TYPE_CLASSIFICATION,
    CE_NEURAL_MODEL_TYPE_GENERATION,
    CE_NEURAL_MODEL_TYPE_CUSTOM
} ce_neural_model_type_t;

typedef enum {
    CE_NEURAL_DEVICE_CPU = 0,
    CE_NEURAL_DEVICE_CUDA,
    CE_NEURAL_DEVICE_OPENVINO,
    CE_NEURAL_DEVICE_TENSORRT
} ce_neural_device_t;

typedef enum {
    CE_NEURAL_PRECISION_FP32 = 0,
    CE_NEURAL_PRECISION_FP16,
    CE_NEURAL_PRECISION_INT8
} ce_neural_precision_t;

/* ============================================================================
 * Neural Model Structure
 * ============================================================================ */

typedef struct {
    char name[64];                  /* Model name */
    char *model_path;               /* Path to ONNX model file */
    ce_neural_model_type_t type;    /* Model type */
    ce_neural_device_t device;      /* Execution device */
    ce_neural_precision_t precision; /* Precision mode */
    
    /* Model metadata */
    size_t input_size;              /* Input tensor size */
    size_t output_size;             /* Output tensor size */
    size_t batch_size;              /* Batch size */
    
    /* ONNX Runtime objects */
    void *session;                  /* ONNX Runtime session */
    void *session_options;          /* Session options */
    void *run_options;              /* Run options */
    
    /* Performance statistics */
    uint64_t inference_count;
    double total_inference_time;
    double avg_inference_time;
    double max_inference_time;
    
    bool is_loaded;                 /* Whether model is loaded */
} ce_neural_model_t;

/* ============================================================================
 * Neural Engine Structure
 * ============================================================================ */

typedef struct ce_neural_engine {
    ce_neural_model_t models[CE_NEURAL_MAX_MODELS];
    size_t model_count;
    
    /* Global configuration */
    ce_neural_device_t default_device;
    ce_neural_precision_t default_precision;
    size_t default_batch_size;
    int num_threads;
    
    /* Performance monitoring */
    uint64_t total_inferences;
    double total_engine_time;
    double avg_engine_time;
    
    /* Synchronization */
    pthread_mutex_t mutex;
    
    /* Callbacks */
    void (*on_model_loaded)(const ce_neural_model_t *model, void *user_data);
    void (*on_inference_complete)(const ce_neural_model_t *model, 
                                 const float *input, const float *output, 
                                 double inference_time, void *user_data);
    void *callback_user_data;
} ce_neural_engine_t;

/* ============================================================================
 * Neural Engine API
 * ============================================================================ */

/**
 * Initialize neural engine
 * @param device Default execution device
 * @param precision Default precision mode
 * @param num_threads Number of threads for CPU execution
 * @return Neural engine instance or NULL on error
 */
ce_neural_engine_t *ce_neural_engine_create(ce_neural_device_t device,
                                           ce_neural_precision_t precision,
                                           int num_threads);

/**
 * Free neural engine
 * @param engine Neural engine instance
 */
void ce_neural_engine_free(ce_neural_engine_t *engine);

/**
 * Load ONNX model
 * @param engine Neural engine instance
 * @param name Model name
 * @param model_path Path to ONNX model file
 * @param type Model type
 * @return Model handle or NULL on error
 */
ce_neural_model_t *ce_neural_engine_load_model(ce_neural_engine_t *engine,
                                               const char *name,
                                               const char *model_path,
                                               ce_neural_model_type_t type);

/**
 * Unload model
 * @param engine Neural engine instance
 * @param model_name Model name to unload
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_engine_unload_model(ce_neural_engine_t *engine,
                                         const char *model_name);

/**
 * Run inference on model
 * @param engine Neural engine instance
 * @param model_name Model name
 * @param input Input tensor
 * @param output Output tensor (must be pre-allocated)
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_engine_infer(ce_neural_engine_t *engine,
                                  const char *model_name,
                                  const float *input,
                                  float *output);

/**
 * Run batch inference
 * @param engine Neural engine instance
 * @param model_name Model name
 * @param inputs Array of input tensors
 * @param outputs Array of output tensors
 * @param batch_size Batch size
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_engine_infer_batch(ce_neural_engine_t *engine,
                                        const char *model_name,
                                        const float **inputs,
                                        float **outputs,
                                        size_t batch_size);

/**
 * Get model by name
 * @param engine Neural engine instance
 * @param model_name Model name
 * @return Model handle or NULL if not found
 */
ce_neural_model_t *ce_neural_engine_get_model(ce_neural_engine_t *engine,
                                              const char *model_name);

/**
 * Get engine statistics
 * @param engine Neural engine instance
 * @param stats Output statistics structure
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_engine_get_stats(const ce_neural_engine_t *engine, struct {
    size_t loaded_models;
    uint64_t total_inferences;
    double avg_inference_time;
    double max_inference_time;
    double total_engine_time;
} *stats);

/**
 * Set callback functions
 * @param engine Neural engine instance
 * @param on_model_loaded Model loaded callback
 * @param on_inference_complete Inference complete callback
 * @param user_data User data passed to callbacks
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_engine_set_callbacks(ce_neural_engine_t *engine,
                                          void (*on_model_loaded)(const ce_neural_model_t *model, void *user_data),
                                          void (*on_inference_complete)(const ce_neural_model_t *model, 
                                                                       const float *input, const float *output, 
                                                                       double inference_time, void *user_data),
                                          void *user_data);

/* ============================================================================
 * Specialized Neural Functions
 * ============================================================================ */

/**
 * Generate embedding for cognitive item
 * @param engine Neural engine instance
 * @param item Cognitive item
 * @param embedding Output embedding (must be pre-allocated)
 * @param embedding_dim Embedding dimension
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_generate_embedding(ce_neural_engine_t *engine,
                                        const ce_item_t *item,
                                        float *embedding,
                                        size_t embedding_dim);

/**
 * Classify cognitive item
 * @param engine Neural engine instance
 * @param item Cognitive item
 * @param probabilities Output class probabilities
 * @param num_classes Number of classes
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_classify_item(ce_neural_engine_t *engine,
                                   const ce_item_t *item,
                                   float *probabilities,
                                   size_t num_classes);

/**
 * Predict next cognitive state
 * @param engine Neural engine instance
 * @param current_state Current cognitive state
 * @param predicted_state Output predicted state
 * @param state_dim State dimension
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_predict_state(ce_neural_engine_t *engine,
                                   const float *current_state,
                                   float *predicted_state,
                                   size_t state_dim);

/**
 * Generate association between items
 * @param engine Neural engine instance
 * @param item1 First item
 * @param item2 Second item
 * @param association_strength Output association strength
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_compute_association(ce_neural_engine_t *engine,
                                         const ce_item_t *item1,
                                         const ce_item_t *item2,
                                         float *association_strength);

/**
 * Pattern recognition in cognitive sequence
 * @param engine Neural engine instance
 * @param sequence Input sequence
 * @param sequence_length Sequence length
 * @param pattern_confidence Output pattern confidence
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_recognize_pattern(ce_neural_engine_t *engine,
                                       const ce_item_t **sequence,
                                       size_t sequence_length,
                                       float *pattern_confidence);

/* ============================================================================
 * Model Management
 * ============================================================================ */

/**
 * Create model from scratch (for custom models)
 * @param engine Neural engine instance
 * @param name Model name
 * @param type Model type
 * @param input_size Input tensor size
 * @param output_size Output tensor size
 * @return Model handle or NULL on error
 */
ce_neural_model_t *ce_neural_create_model(ce_neural_engine_t *engine,
                                          const char *name,
                                          ce_neural_model_type_t type,
                                          size_t input_size,
                                          size_t output_size);

/**
 * Save model to file
 * @param model Model to save
 * @param filepath Output file path
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_save_model(const ce_neural_model_t *model,
                                const char *filepath);

/**
 * Load model from file
 * @param engine Neural engine instance
 * @param filepath Model file path
 * @return Model handle or NULL on error
 */
ce_neural_model_t *ce_neural_load_model_from_file(ce_neural_engine_t *engine,
                                                  const char *filepath);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Get device capabilities
 * @param device Device type
 * @param capabilities Output capabilities structure
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_get_device_capabilities(ce_neural_device_t device, struct {
    bool available;
    size_t max_memory;
    int compute_capability;
    char device_name[256];
} *capabilities);

/**
 * Convert cognitive item to tensor
 * @param item Cognitive item
 * @param tensor Output tensor
 * @param max_size Maximum tensor size
 * @return Actual tensor size
 */
size_t ce_neural_item_to_tensor(const ce_item_t *item, float *tensor, size_t max_size);

/**
 * Convert tensor to cognitive item
 * @param tensor Input tensor
 * @param tensor_size Tensor size
 * @param item_type Item type
 * @param confidence Confidence level
 * @return Cognitive item or NULL on error
 */
ce_item_t *ce_neural_tensor_to_item(const float *tensor, size_t tensor_size,
                                    ce_item_type_t item_type, float confidence);

/**
 * Benchmark model performance
 * @param model Model to benchmark
 * @param num_iterations Number of benchmark iterations
 * @param stats Output benchmark statistics
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_benchmark_model(ce_neural_model_t *model,
                                     size_t num_iterations,
                                     struct {
                                         double avg_inference_time;
                                         double min_inference_time;
                                         double max_inference_time;
                                         double throughput;
                                     } *stats);

#endif /* CE_NEURAL_ENGINE_H */
