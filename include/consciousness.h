/**
 * Consciousness Emulator (CE) - Main Header
 * 
 * A modular AI microkernel implementing Global Workspace Theory,
 * working memory, long-term memory, and self-modeling capabilities.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 * 
 * This is the main entry point for the Consciousness Emulator library.
 * It provides a unified interface to all core cognitive modules.
 */

#ifndef CONSCIOUSNESS_H
#define CONSCIOUSNESS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Core Types and Constants
 * ============================================================================ */

#define CE_VERSION_MAJOR 1
#define CE_VERSION_MINOR 1
#define CE_VERSION_PATCH 0

#define CE_MAX_MODULES 32
#define CE_MAX_ITEMS 1024
#define CE_DEFAULT_WM_CAPACITY 16
#define CE_DEFAULT_TICK_HZ 50.0
#define CE_MAX_EMBEDDING_DIM 512
#define CE_MAX_STRING_LENGTH 1024

/* Error codes */
typedef enum {
    CE_SUCCESS = 0,
    CE_ERROR_NULL_POINTER = -1,
    CE_ERROR_INVALID_PARAM = -2,
    CE_ERROR_OUT_OF_MEMORY = -3,
    CE_ERROR_MODULE_NOT_FOUND = -4,
    CE_ERROR_INVALID_STATE = -5,
    CE_ERROR_SERIALIZATION = -6,
    CE_ERROR_DESERIALIZATION = -7,
    CE_ERROR_IO = -8,
    CE_ERROR_MATH = -9,
    CE_ERROR_UNKNOWN = -100
} ce_error_t;

/* Item types for the cognitive workspace */
typedef enum {
    CE_ITEM_TYPE_SENSORY = 0,
    CE_ITEM_TYPE_MEMORY,
    CE_ITEM_TYPE_GOAL,
    CE_ITEM_TYPE_BELIEF,
    CE_ITEM_TYPE_ACTION,
    CE_ITEM_TYPE_QUESTION,
    CE_ITEM_TYPE_ANSWER,
    CE_ITEM_TYPE_PREDICTION,
    CE_ITEM_TYPE_EMOTION,
    CE_ITEM_TYPE_META,
    CE_ITEM_TYPE_CUSTOM = 100
} ce_item_type_t;

/* Core Item structure - the fundamental unit of information */
typedef struct {
    uint64_t id;                    /* Unique identifier */
    ce_item_type_t type;            /* Type of cognitive item */
    double timestamp;               /* Creation time */
    double last_accessed;           /* Last access time */
    float confidence;               /* Confidence level [0.0, 1.0] */
    float saliency;                 /* Attention saliency [0.0, 1.0] */
    float *embedding;               /* Vector representation */
    size_t embedding_dim;           /* Embedding dimension */
    char *content;                  /* Textual content */
    char *metadata;                 /* JSON metadata */
    void *payload;                  /* Type-specific data */
    struct ce_item *provenance;     /* Source item (for reasoning chains) */
    struct ce_item *next;           /* Linked list for chains */
} ce_item_t;

/* Item list for collections */
typedef struct {
    ce_item_t **items;              /* Array of item pointers */
    size_t count;                   /* Current count */
    size_t capacity;                /* Allocated capacity */
    float total_saliency;           /* Sum of all saliencies */
} ce_item_list_t;

/* ============================================================================
 * Core System Interface
 * ============================================================================ */

/**
 * Initialize the Consciousness Emulator system
 * @param tick_hz Frequency of the main cognitive loop
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_init(double tick_hz);

/**
 * Shutdown the Consciousness Emulator system
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_shutdown(void);

/**
 * Run the main cognitive loop for a specified duration
 * @param duration_seconds How long to run (0 = run indefinitely)
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_run(double duration_seconds);

/**
 * Process a single cognitive tick
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_tick(void);

/**
 * Get current system timestamp
 * @return Current timestamp in seconds since epoch
 */
double ce_get_timestamp(void);

/* ============================================================================
 * Item Management
 * ============================================================================ */

/**
 * Create a new cognitive item
 * @param type Item type
 * @param content Textual content
 * @param confidence Confidence level
 * @return New item or NULL on error
 */
ce_item_t *ce_item_create(ce_item_type_t type, const char *content, float confidence);

/**
 * Create an item with embedding
 * @param type Item type
 * @param content Textual content
 * @param embedding Vector embedding
 * @param embedding_dim Embedding dimension
 * @param confidence Confidence level
 * @return New item or NULL on error
 */
ce_item_t *ce_item_create_with_embedding(ce_item_type_t type, const char *content,
                                        const float *embedding, size_t embedding_dim,
                                        float confidence);

/**
 * Clone an item
 * @param item Source item
 * @return Cloned item or NULL on error
 */
ce_item_t *ce_item_clone(const ce_item_t *item);

/**
 * Update item saliency
 * @param item Item to update
 * @param saliency New saliency value
 */
void ce_item_update_saliency(ce_item_t *item, float saliency);

/**
 * Update item confidence
 * @param item Item to update
 * @param confidence New confidence value
 */
void ce_item_update_confidence(ce_item_t *item, float confidence);

/**
 * Free an item
 * @param item Item to free
 */
void ce_item_free(ce_item_t *item);

/**
 * Create an item list
 * @param initial_capacity Initial capacity
 * @return New item list or NULL on error
 */
ce_item_list_t *ce_item_list_create(size_t initial_capacity);

/**
 * Add item to list
 * @param list Item list
 * @param item Item to add
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_item_list_add(ce_item_list_t *list, ce_item_t *item);

/**
 * Remove item from list
 * @param list Item list
 * @param item Item to remove
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_item_list_remove(ce_item_list_t *list, ce_item_t *item);

/**
 * Get top-k items by saliency
 * @param list Item list
 * @param k Number of items to return
 * @param result Output array (must be pre-allocated)
 * @return Number of items returned
 */
size_t ce_item_list_topk(const ce_item_list_t *list, size_t k, ce_item_t **result);

/**
 * Free an item list
 * @param list Item list to free
 */
void ce_item_list_free(ce_item_list_t *list);

/* ============================================================================
 * Working Memory Interface
 * ============================================================================ */

/**
 * Working Memory handle (opaque)
 */
typedef struct ce_working_memory ce_working_memory_t;

/**
 * Create working memory
 * @param capacity Maximum number of items
 * @return Working memory handle or NULL on error
 */
ce_working_memory_t *ce_wm_create(size_t capacity);

/**
 * Add item to working memory
 * @param wm Working memory handle
 * @param item Item to add
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_add(ce_working_memory_t *wm, ce_item_t *item);

/**
 * Get current working memory items
 * @param wm Working memory handle
 * @return Item list (do not free, owned by WM)
 */
const ce_item_list_t *ce_wm_get_items(const ce_working_memory_t *wm);

/**
 * Update working memory (decay, consolidation)
 * @param wm Working memory handle
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_update(ce_working_memory_t *wm);

/**
 * Free working memory
 * @param wm Working memory handle
 */
void ce_wm_free(ce_working_memory_t *wm);

/* ============================================================================
 * Global Workspace Interface
 * ============================================================================ */

/**
 * Global Workspace handle (opaque)
 */
typedef struct ce_workspace ce_workspace_t;

/**
 * Create global workspace
 * @param wm Working memory handle
 * @param threshold Broadcast threshold
 * @return Workspace handle or NULL on error
 */
ce_workspace_t *ce_workspace_create(ce_working_memory_t *wm, float threshold);

/**
 * Process workspace (attention, arbitration, broadcast)
 * @param workspace Workspace handle
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_workspace_process(ce_workspace_t *workspace);

/**
 * Get currently broadcasted items
 * @param workspace Workspace handle
 * @return Item list (do not free, owned by workspace)
 */
const ce_item_list_t *ce_workspace_get_broadcast(const ce_workspace_t *workspace);

/**
 * Free workspace
 * @param workspace Workspace handle
 */
void ce_workspace_free(ce_workspace_t *workspace);

/* ============================================================================
 * Long-Term Memory Interface
 * ============================================================================ */

/**
 * Long-Term Memory handle (opaque)
 */
typedef struct ce_long_term_memory ce_long_term_memory_t;

/**
 * Create long-term memory
 * @param embedding_dim Embedding dimension
 * @param max_episodes Maximum number of episodes
 * @return LTM handle or NULL on error
 */
ce_long_term_memory_t *ce_ltm_create(size_t embedding_dim, size_t max_episodes);

/**
 * Store episode in LTM
 * @param ltm LTM handle
 * @param item Item to store
 * @param context Additional context
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_store_episode(ce_long_term_memory_t *ltm, const ce_item_t *item, const char *context);

/**
 * Search LTM by similarity
 * @param ltm LTM handle
 * @param query_embedding Query vector
 * @param k Number of results
 * @param results Output array (must be pre-allocated)
 * @return Number of results found
 */
size_t ce_ltm_search(ce_long_term_memory_t *ltm, const float *query_embedding, 
                     size_t k, ce_item_t **results);

/**
 * Consolidate memories
 * @param ltm LTM handle
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_consolidate(ce_long_term_memory_t *ltm);

/**
 * Free LTM
 * @param ltm LTM handle
 */
void ce_ltm_free(ce_long_term_memory_t *ltm);

/* ============================================================================
 * Reasoning Engine Interface
 * ============================================================================ */

/**
 * Reasoning Engine handle (opaque)
 */
typedef struct ce_reasoner ce_reasoner_t;

/**
 * Create reasoning engine
 * @return Reasoner handle or NULL on error
 */
ce_reasoner_t *ce_reasoner_create(void);

/**
 * Process reasoning on broadcasted items
 * @param reasoner Reasoner handle
 * @param broadcast_items Items to reason about
 * @return Generated reasoning items
 */
ce_item_list_t *ce_reasoner_process(ce_reasoner_t *reasoner, const ce_item_list_t *broadcast_items);

/**
 * Add reasoning rule
 * @param reasoner Reasoner handle
 * @param rule_name Rule identifier
 * @param condition Condition function
 * @param action Action function
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_reasoner_add_rule(ce_reasoner_t *reasoner, const char *rule_name,
                               bool (*condition)(const ce_item_t *),
                               ce_item_t *(*action)(const ce_item_t *));

/**
 * Free reasoner
 * @param reasoner Reasoner handle
 */
void ce_reasoner_free(ce_reasoner_t *reasoner);

/* ============================================================================
 * Self-Model Interface
 * ============================================================================ */

/**
 * Self-Model handle (opaque)
 */
typedef struct ce_self_model ce_self_model_t;

/**
 * Create self-model
 * @return Self-model handle or NULL on error
 */
ce_self_model_t *ce_self_model_create(void);

/**
 * Update self-model with current state
 * @param self_model Self-model handle
 * @param workspace Current workspace state
 * @param recent_thoughts Recent reasoning items
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_self_model_update(ce_self_model_t *self_model, 
                               const ce_workspace_t *workspace,
                               const ce_item_list_t *recent_thoughts);

/**
 * Get self-model summary
 * @param self_model Self-model handle
 * @return Human-readable summary (caller must free)
 */
char *ce_self_model_get_summary(const ce_self_model_t *self_model);

/**
 * Get self-model explanation for decision
 * @param self_model Self-model handle
 * @param item Item to explain
 * @return Explanation string (caller must free)
 */
char *ce_self_model_explain_decision(const ce_self_model_t *self_model, const ce_item_t *item);

/**
 * Free self-model
 * @param self_model Self-model handle
 */
void ce_self_model_free(ce_self_model_t *self_model);

/* ============================================================================
 * IO and Control Interface
 * ============================================================================ */

/**
 * Input handler function type
 */
typedef ce_item_t *(*ce_input_handler_t)(const char *input, void *user_data);

/**
 * Output handler function type
 */
typedef void (*ce_output_handler_t)(const char *output, void *user_data);

/**
 * Register input handler
 * @param handler Input handler function
 * @param user_data User data passed to handler
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_io_register_input_handler(ce_input_handler_t handler, void *user_data);

/**
 * Register output handler
 * @param handler Output handler function
 * @param user_data User data passed to handler
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_io_register_output_handler(ce_output_handler_t handler, void *user_data);

/**
 * Send input to the system
 * @param input Input string
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_io_send_input(const char *input);

/**
 * Get system output
 * @param output Buffer for output (must be pre-allocated)
 * @param max_length Maximum buffer length
 * @return Number of characters written
 */
size_t ce_io_get_output(char *output, size_t max_length);

/* ============================================================================
 * v1.1 Advanced Features
 * ============================================================================ */

/**
 * Neural Network Engine (v1.1)
 */
typedef struct ce_neural_engine ce_neural_engine_t;

/**
 * Create neural engine
 * @param device Execution device (0=CPU, 1=CUDA, 2=OpenVINO, 3=TensorRT)
 * @param precision Precision mode (0=FP32, 1=FP16, 2=INT8)
 * @param num_threads Number of threads for CPU execution
 * @return Neural engine instance or NULL on error
 */
ce_neural_engine_t *ce_neural_engine_create(int device, int precision, int num_threads);

/**
 * Load ONNX model
 * @param engine Neural engine instance
 * @param name Model name
 * @param model_path Path to ONNX model file
 * @param type Model type (0=pattern, 1=association, 2=prediction, 3=embedding, 4=classification, 5=generation, 6=custom)
 * @return Model handle or NULL on error
 */
void *ce_neural_engine_load_model(ce_neural_engine_t *engine, const char *name,
                                  const char *model_path, int type);

/**
 * Run neural inference
 * @param engine Neural engine instance
 * @param model_name Model name
 * @param input Input tensor
 * @param output Output tensor (must be pre-allocated)
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_neural_engine_infer(ce_neural_engine_t *engine, const char *model_name,
                                  const float *input, float *output);

/**
 * Free neural engine
 * @param engine Neural engine instance
 */
void ce_neural_engine_free(ce_neural_engine_t *engine);

/**
 * CUDA Acceleration (v1.1)
 */
typedef struct ce_cuda_context ce_cuda_context_t;

/**
 * Initialize CUDA context
 * @return CUDA context or NULL on error
 */
ce_cuda_context_t *ce_cuda_init(void);

/**
 * Check if CUDA is available
 * @return True if CUDA is available
 */
bool ce_cuda_is_available(void);

/**
 * CUDA vector addition
 * @param a First vector (device)
 * @param b Second vector (device)
 * @param c Result vector (device)
 * @param n Vector length
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_vector_add(const float *a, const float *b, float *c, size_t n);

/**
 * CUDA matrix multiplication
 * @param A First matrix (device)
 * @param B Second matrix (device)
 * @param C Result matrix (device)
 * @param m Rows of A
 * @param k Columns of A / Rows of B
 * @param n Columns of B
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_cuda_matrix_multiply(const float *A, const float *B, float *C,
                                   size_t m, size_t k, size_t n);

/**
 * Free CUDA context
 * @param context CUDA context
 */
void ce_cuda_free(ce_cuda_context_t *context);

/**
 * Advanced Reasoning Engine (v1.1)
 */
typedef struct ce_advanced_reasoner ce_advanced_reasoner_t;

/**
 * Create advanced reasoner
 * @param mode Reasoning mode (0=forward, 1=backward, 2=probabilistic, 3=causal, 4=analogical, 5=abductive, 6=deductive, 7=inductive, 8=meta)
 * @param confidence_threshold Confidence threshold
 * @param max_depth Maximum inference depth
 * @return Advanced reasoner instance or NULL on error
 */
ce_advanced_reasoner_t *ce_advanced_reasoner_create(int mode, float confidence_threshold, int max_depth);

/**
 * Add reasoning rule
 * @param reasoner Advanced reasoner instance
 * @param name Rule name
 * @param type Rule type (0=fact, 1=implication, 2=constraint, 3=causal, 4=analogical, 5=probabilistic, 6=temporal, 7=meta)
 * @param premise Rule premise
 * @param conclusion Rule conclusion
 * @param confidence Rule confidence
 * @return Rule ID or 0 on error
 */
uint64_t ce_advanced_reasoner_add_rule(ce_advanced_reasoner_t *reasoner,
                                       const char *name, int type,
                                       const char *premise, const char *conclusion,
                                       float confidence);

/**
 * Perform advanced reasoning
 * @param reasoner Advanced reasoner instance
 * @param context_id Context ID
 * @param mode Reasoning mode
 * @return Generated conclusions
 */
ce_item_list_t *ce_advanced_reasoner_reason(ce_advanced_reasoner_t *reasoner,
                                            uint64_t context_id, int mode);

/**
 * Free advanced reasoner
 * @param reasoner Advanced reasoner instance
 */
void ce_advanced_reasoner_free(ce_advanced_reasoner_t *reasoner);

/**
 * Web Visualization Interface (v1.1)
 */
typedef struct ce_web_interface ce_web_interface_t;

/**
 * Create web interface
 * @param port Server port
 * @param update_interval Update interval in seconds
 * @return Web interface instance or NULL on error
 */
ce_web_interface_t *ce_web_interface_create(int port, double update_interval);

/**
 * Start web server
 * @param web_interface Web interface instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_interface_start(ce_web_interface_t *web_interface);

/**
 * Update visualization data
 * @param web_interface Web interface instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_interface_update_data(ce_web_interface_t *web_interface);

/**
 * Stop web server
 * @param web_interface Web interface instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_interface_stop(ce_web_interface_t *web_interface);

/**
 * Free web interface
 * @param web_interface Web interface instance
 */
void ce_web_interface_free(ce_web_interface_t *web_interface);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Get error string
 * @param error Error code
 * @return Error description
 */
const char *ce_error_string(ce_error_t error);

/**
 * Get version string
 * @return Version string
 */
const char *ce_get_version(void);


/**
 * Generate random embedding
 * @param embedding Output embedding array
 * @param dim Embedding dimension
 * @param seed Random seed
 */
void ce_generate_random_embedding(float *embedding, size_t dim, uint32_t seed);

/**
 * Compute cosine similarity
 * @param a First vector
 * @param b Second vector
 * @param dim Vector dimension
 * @return Cosine similarity [-1.0, 1.0]
 */
float ce_cosine_similarity(const float *a, const float *b, size_t dim);

/**
 * Compute L2 distance
 * @param a First vector
 * @param b Second vector
 * @param dim Vector dimension
 * @return L2 distance
 */
float ce_l2_distance(const float *a, const float *b, size_t dim);

#ifdef __cplusplus
}
#endif

#endif /* CONSCIOUSNESS_H */
