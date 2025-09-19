/**
 * Consciousness Emulator - Kernel Module
 * 
 * Implements the microkernel scheduler and message bus for the CE system.
 * This is the core orchestrator that manages all cognitive modules.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#ifndef CE_KERNEL_H
#define CE_KERNEL_H

#include "../../include/consciousness.h"
#include <pthread.h>
#include <stdatomic.h>

/* ============================================================================
 * Kernel Configuration
 * ============================================================================ */

#define CE_KERNEL_MAX_MODULES 32
#define CE_KERNEL_MAX_MESSAGES 1024
#define CE_KERNEL_DEFAULT_TICK_HZ 50.0
#define CE_KERNEL_MESSAGE_TIMEOUT 1.0

/* ============================================================================
 * Module Types and Interfaces
 * ============================================================================ */

typedef enum {
    CE_MODULE_TYPE_SENSORY = 0,
    CE_MODULE_TYPE_WORKING_MEMORY,
    CE_MODULE_TYPE_WORKSPACE,
    CE_MODULE_TYPE_LONG_TERM_MEMORY,
    CE_MODULE_TYPE_REASONER,
    CE_MODULE_TYPE_SELF_MODEL,
    CE_MODULE_TYPE_IO,
    CE_MODULE_TYPE_CUSTOM
} ce_module_type_t;

typedef struct ce_module ce_module_t;

/* Module callback function types */
typedef ce_error_t (*ce_module_init_func_t)(ce_module_t *module, void *config);
typedef ce_error_t (*ce_module_tick_func_t)(ce_module_t *module, double timestamp);
typedef ce_error_t (*ce_module_shutdown_func_t)(ce_module_t *module);
typedef ce_error_t (*ce_module_broadcast_func_t)(ce_module_t *module, const ce_item_t *item);
typedef ce_error_t (*ce_module_serialize_func_t)(const ce_module_t *module, char **data, size_t *size);
typedef ce_error_t (*ce_module_deserialize_func_t)(ce_module_t *module, const char *data, size_t size);

/* Module structure */
struct ce_module {
    char name[64];                          /* Module name */
    ce_module_type_t type;                  /* Module type */
    void *context;                          /* Module-specific context */
    
    /* Callback functions */
    ce_module_init_func_t init_func;
    ce_module_tick_func_t tick_func;
    ce_module_shutdown_func_t shutdown_func;
    ce_module_broadcast_func_t broadcast_func;
    ce_module_serialize_func_t serialize_func;
    ce_module_deserialize_func_t deserialize_func;
    
    /* State */
    bool initialized;
    bool active;
    double last_tick;
    double tick_interval;                   /* Minimum time between ticks */
    
    /* Statistics */
    uint64_t tick_count;
    uint64_t broadcast_count;
    double total_tick_time;
    double max_tick_time;
};

/* ============================================================================
 * Message System
 * ============================================================================ */

typedef enum {
    CE_MESSAGE_TYPE_BROADCAST = 0,
    CE_MESSAGE_TYPE_REQUEST,
    CE_MESSAGE_TYPE_RESPONSE,
    CE_MESSAGE_TYPE_EVENT,
    CE_MESSAGE_TYPE_CONTROL
} ce_message_type_t;

typedef struct {
    uint64_t id;
    ce_message_type_t type;
    double timestamp;
    char source_module[64];
    char target_module[64];
    ce_item_t *payload;
    void *user_data;
    size_t user_data_size;
} ce_message_t;

typedef struct {
    ce_message_t *messages;
    size_t count;
    size_t capacity;
    size_t head;
    size_t tail;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
} ce_message_queue_t;

/* ============================================================================
 * Kernel State
 * ============================================================================ */

typedef struct {
    /* Configuration */
    double tick_hz;
    double tick_interval;
    bool running;
    bool paused;
    
    /* Modules */
    ce_module_t modules[CE_KERNEL_MAX_MODULES];
    size_t module_count;
    
    /* Message system */
    ce_message_queue_t message_queue;
    pthread_t message_thread;
    
    /* Main loop */
    pthread_t main_thread;
    atomic_bool should_stop;
    
    /* Timing */
    double start_time;
    double last_tick_time;
    uint64_t total_ticks;
    
    /* Statistics */
    double total_runtime;
    double avg_tick_time;
    double max_tick_time;
    
    /* Synchronization */
    pthread_mutex_t kernel_mutex;
    pthread_cond_t kernel_condition;
    
    /* Global state */
    ce_working_memory_t *global_wm;
    ce_workspace_t *global_workspace;
    ce_long_term_memory_t *global_ltm;
    ce_reasoner_t *global_reasoner;
    ce_self_model_t *global_self_model;
} ce_kernel_t;

/* ============================================================================
 * Kernel API
 * ============================================================================ */

/**
 * Initialize the kernel
 * @param tick_hz Frequency of the main loop
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_kernel_init(double tick_hz);

/**
 * Shutdown the kernel
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_kernel_shutdown(void);

/**
 * Register a module with the kernel
 * @param name Module name
 * @param type Module type
 * @param context Module context
 * @param init_func Initialization function
 * @param tick_func Tick function
 * @param shutdown_func Shutdown function
 * @param broadcast_func Broadcast handler
 * @param serialize_func Serialization function
 * @param deserialize_func Deserialization function
 * @return Module handle or NULL on error
 */
ce_module_t *ce_kernel_register_module(const char *name, ce_module_type_t type,
                                      void *context,
                                      ce_module_init_func_t init_func,
                                      ce_module_tick_func_t tick_func,
                                      ce_module_shutdown_func_t shutdown_func,
                                      ce_module_broadcast_func_t broadcast_func,
                                      ce_module_serialize_func_t serialize_func,
                                      ce_module_deserialize_func_t deserialize_func);

/**
 * Unregister a module
 * @param module Module to unregister
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_kernel_unregister_module(ce_module_t *module);

/**
 * Start the kernel main loop
 * @param duration_seconds Duration to run (0 = indefinitely)
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_kernel_start(double duration_seconds);

/**
 * Stop the kernel
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_kernel_stop(void);

/**
 * Pause the kernel
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_kernel_pause(void);

/**
 * Resume the kernel
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_kernel_resume(void);

/**
 * Process a single tick
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_kernel_tick(void);

/**
 * Send a message
 * @param message Message to send
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_kernel_send_message(const ce_message_t *message);

/**
 * Broadcast an item to all modules
 * @param item Item to broadcast
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_kernel_broadcast_item(const ce_item_t *item);

/**
 * Get kernel statistics
 * @param stats Output statistics structure
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_kernel_get_stats(struct {
    double total_runtime;
    uint64_t total_ticks;
    double avg_tick_time;
    double max_tick_time;
    size_t active_modules;
    size_t message_queue_size;
} *stats);

/**
 * Get global working memory
 * @return Global working memory handle
 */
ce_working_memory_t *ce_kernel_get_global_wm(void);

/**
 * Get global workspace
 * @return Global workspace handle
 */
ce_workspace_t *ce_kernel_get_global_workspace(void);

/**
 * Get global long-term memory
 * @return Global LTM handle
 */
ce_long_term_memory_t *ce_kernel_get_global_ltm(void);

/**
 * Get global reasoner
 * @return Global reasoner handle
 */
ce_reasoner_t *ce_kernel_get_global_reasoner(void);

/**
 * Get global self-model
 * @return Global self-model handle
 */
ce_self_model_t *ce_kernel_get_global_self_model(void);

/* ============================================================================
 * Message Queue API
 * ============================================================================ */

/**
 * Create a message queue
 * @param capacity Maximum number of messages
 * @return Message queue or NULL on error
 */
ce_message_queue_t *ce_message_queue_create(size_t capacity);

/**
 * Destroy a message queue
 * @param queue Message queue to destroy
 */
void ce_message_queue_destroy(ce_message_queue_t *queue);

/**
 * Enqueue a message
 * @param queue Message queue
 * @param message Message to enqueue
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_message_queue_enqueue(ce_message_queue_t *queue, const ce_message_t *message);

/**
 * Dequeue a message
 * @param queue Message queue
 * @param message Output message
 * @param timeout_seconds Timeout in seconds (0 = non-blocking)
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_message_queue_dequeue(ce_message_queue_t *queue, ce_message_t *message, double timeout_seconds);

/**
 * Get queue size
 * @param queue Message queue
 * @return Number of messages in queue
 */
size_t ce_message_queue_size(const ce_message_queue_t *queue);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Get current kernel instance
 * @return Kernel instance or NULL if not initialized
 */
ce_kernel_t *ce_kernel_get_instance(void);

/**
 * Get module by name
 * @param name Module name
 * @return Module or NULL if not found
 */
ce_module_t *ce_kernel_get_module(const char *name);

/**
 * Get module by type
 * @param type Module type
 * @return Module or NULL if not found
 */
ce_module_t *ce_kernel_get_module_by_type(ce_module_type_t type);

/**
 * Create a message
 * @param type Message type
 * @param source Source module name
 * @param target Target module name
 * @param payload Message payload
 * @return New message or NULL on error
 */
ce_message_t *ce_message_create(ce_message_type_t type, const char *source, 
                               const char *target, ce_item_t *payload);

/**
 * Free a message
 * @param message Message to free
 */
void ce_message_free(ce_message_t *message);

#endif /* CE_KERNEL_H */
