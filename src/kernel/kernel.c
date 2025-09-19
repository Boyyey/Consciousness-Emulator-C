/**
 * Consciousness Emulator - Kernel Implementation
 * 
 * The microkernel scheduler and message bus implementation.
 * This is the heart of the CE system, orchestrating all cognitive modules.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "kernel.h"
#include "../utils/arena.h"
#include "../utils/math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <errno.h>

/* ============================================================================
 * Global Kernel Instance
 * ============================================================================ */

static ce_kernel_t *g_kernel = NULL;
static pthread_mutex_t g_kernel_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ============================================================================
 * Internal Functions
 * ============================================================================ */

/**
 * Get current time in seconds with high precision
 */
static double get_current_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

/**
 * Message processing thread
 */
static void *message_thread_func(void *arg) {
    ce_kernel_t *kernel = (ce_kernel_t *)arg;
    ce_message_t message;
    
    while (!atomic_load(&kernel->should_stop)) {
        ce_error_t result = ce_message_queue_dequeue(&kernel->message_queue, &message, 0.1);
        
        if (result == CE_SUCCESS) {
            /* Process broadcast messages */
            if (message.type == CE_MESSAGE_TYPE_BROADCAST) {
                for (size_t i = 0; i < kernel->module_count; i++) {
                    ce_module_t *module = &kernel->modules[i];
                    if (module->active && module->broadcast_func) {
                        module->broadcast_func(module, message.payload);
                        module->broadcast_count++;
                    }
                }
            }
            
            /* Free message payload if it was allocated */
            if (message.payload) {
                ce_item_free(message.payload);
            }
        }
    }
    
    return NULL;
}

/**
 * Main kernel loop thread
 */
static void *main_loop_thread_func(void *arg) {
    ce_kernel_t *kernel = (ce_kernel_t *)arg;
    double target_interval = 1.0 / kernel->tick_hz;
    double next_tick_time = get_current_time() + target_interval;
    
    while (!atomic_load(&kernel->should_stop)) {
        double current_time = get_current_time();
        
        if (current_time >= next_tick_time) {
            double tick_start = get_current_time();
            
            /* Process kernel tick */
            ce_kernel_tick();
            
            double tick_end = get_current_time();
            double tick_duration = tick_end - tick_start;
            
            /* Update statistics */
            kernel->total_ticks++;
            kernel->total_tick_time += tick_duration;
            kernel->avg_tick_time = kernel->total_tick_time / kernel->total_ticks;
            
            if (tick_duration > kernel->max_tick_time) {
                kernel->max_tick_time = tick_duration;
            }
            
            /* Update timing */
            kernel->last_tick_time = tick_end;
            next_tick_time = tick_end + target_interval;
        } else {
            /* Sleep until next tick */
            double sleep_time = next_tick_time - current_time;
            if (sleep_time > 0) {
                usleep((useconds_t)(sleep_time * 1000000));
            }
        }
    }
    
    return NULL;
}

/**
 * Initialize global cognitive modules
 */
static ce_error_t init_global_modules(ce_kernel_t *kernel) {
    /* Create global working memory */
    kernel->global_wm = ce_wm_create(CE_DEFAULT_WM_CAPACITY);
    if (!kernel->global_wm) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Create global workspace */
    kernel->global_workspace = ce_workspace_create(kernel->global_wm, 0.5f);
    if (!kernel->global_workspace) {
        ce_wm_free(kernel->global_wm);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Create global long-term memory */
    kernel->global_ltm = ce_ltm_create(CE_MAX_EMBEDDING_DIM, 10000);
    if (!kernel->global_ltm) {
        ce_workspace_free(kernel->global_workspace);
        ce_wm_free(kernel->global_wm);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Create global reasoner */
    kernel->global_reasoner = ce_reasoner_create();
    if (!kernel->global_reasoner) {
        ce_ltm_free(kernel->global_ltm);
        ce_workspace_free(kernel->global_workspace);
        ce_wm_free(kernel->global_wm);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Create global self-model */
    kernel->global_self_model = ce_self_model_create();
    if (!kernel->global_self_model) {
        ce_reasoner_free(kernel->global_reasoner);
        ce_ltm_free(kernel->global_ltm);
        ce_workspace_free(kernel->global_workspace);
        ce_wm_free(kernel->global_wm);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    return CE_SUCCESS;
}

/**
 * Cleanup global cognitive modules
 */
static void cleanup_global_modules(ce_kernel_t *kernel) {
    if (kernel->global_self_model) {
        ce_self_model_free(kernel->global_self_model);
        kernel->global_self_model = NULL;
    }
    
    if (kernel->global_reasoner) {
        ce_reasoner_free(kernel->global_reasoner);
        kernel->global_reasoner = NULL;
    }
    
    if (kernel->global_ltm) {
        ce_ltm_free(kernel->global_ltm);
        kernel->global_ltm = NULL;
    }
    
    if (kernel->global_workspace) {
        ce_workspace_free(kernel->global_workspace);
        kernel->global_workspace = NULL;
    }
    
    if (kernel->global_wm) {
        ce_wm_free(kernel->global_wm);
        kernel->global_wm = NULL;
    }
}

/* ============================================================================
 * Public API Implementation
 * ============================================================================ */

ce_error_t ce_kernel_init(double tick_hz) {
    pthread_mutex_lock(&g_kernel_mutex);
    
    if (g_kernel) {
        pthread_mutex_unlock(&g_kernel_mutex);
        return CE_ERROR_INVALID_STATE;
    }
    
    /* Allocate kernel structure */
    g_kernel = calloc(1, sizeof(ce_kernel_t));
    if (!g_kernel) {
        pthread_mutex_unlock(&g_kernel_mutex);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Initialize configuration */
    g_kernel->tick_hz = tick_hz;
    g_kernel->tick_interval = 1.0 / tick_hz;
    g_kernel->start_time = get_current_time();
    
    /* Initialize synchronization primitives */
    if (pthread_mutex_init(&g_kernel->kernel_mutex, NULL) != 0) {
        free(g_kernel);
        g_kernel = NULL;
        pthread_mutex_unlock(&g_kernel_mutex);
        return CE_ERROR_UNKNOWN;
    }
    
    if (pthread_cond_init(&g_kernel->kernel_condition, NULL) != 0) {
        pthread_mutex_destroy(&g_kernel->kernel_mutex);
        free(g_kernel);
        g_kernel = NULL;
        pthread_mutex_unlock(&g_kernel_mutex);
        return CE_ERROR_UNKNOWN;
    }
    
    /* Initialize message queue */
    g_kernel->message_queue = *ce_message_queue_create(CE_KERNEL_MAX_MESSAGES);
    if (!g_kernel->message_queue.messages) {
        pthread_cond_destroy(&g_kernel->kernel_condition);
        pthread_mutex_destroy(&g_kernel->kernel_mutex);
        free(g_kernel);
        g_kernel = NULL;
        pthread_mutex_unlock(&g_kernel_mutex);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Initialize global cognitive modules */
    ce_error_t result = init_global_modules(g_kernel);
    if (result != CE_SUCCESS) {
        ce_message_queue_destroy(&g_kernel->message_queue);
        pthread_cond_destroy(&g_kernel->kernel_condition);
        pthread_mutex_destroy(&g_kernel->kernel_mutex);
        free(g_kernel);
        g_kernel = NULL;
        pthread_mutex_unlock(&g_kernel_mutex);
        return result;
    }
    
    pthread_mutex_unlock(&g_kernel_mutex);
    return CE_SUCCESS;
}

ce_error_t ce_kernel_shutdown(void) {
    pthread_mutex_lock(&g_kernel_mutex);
    
    if (!g_kernel) {
        pthread_mutex_unlock(&g_kernel_mutex);
        return CE_ERROR_INVALID_STATE;
    }
    
    /* Stop the kernel if running */
    if (g_kernel->running) {
        ce_kernel_stop();
    }
    
    /* Shutdown all modules */
    for (size_t i = 0; i < g_kernel->module_count; i++) {
        ce_module_t *module = &g_kernel->modules[i];
        if (module->initialized && module->shutdown_func) {
            module->shutdown_func(module);
        }
    }
    
    /* Cleanup global modules */
    cleanup_global_modules(g_kernel);
    
    /* Destroy message queue */
    ce_message_queue_destroy(&g_kernel->message_queue);
    
    /* Destroy synchronization primitives */
    pthread_cond_destroy(&g_kernel->kernel_condition);
    pthread_mutex_destroy(&g_kernel->kernel_mutex);
    
    /* Free kernel structure */
    free(g_kernel);
    g_kernel = NULL;
    
    pthread_mutex_unlock(&g_kernel_mutex);
    return CE_SUCCESS;
}

ce_error_t ce_kernel_start(double duration_seconds) {
    if (!g_kernel) {
        return CE_ERROR_INVALID_STATE;
    }
    
    if (g_kernel->running) {
        return CE_ERROR_INVALID_STATE;
    }
    
    /* Initialize all modules */
    for (size_t i = 0; i < g_kernel->module_count; i++) {
        ce_module_t *module = &g_kernel->modules[i];
        if (!module->initialized && module->init_func) {
            ce_error_t result = module->init_func(module, module->context);
            if (result != CE_SUCCESS) {
                /* Shutdown already initialized modules */
                for (size_t j = 0; j < i; j++) {
                    if (g_kernel->modules[j].initialized && g_kernel->modules[j].shutdown_func) {
                        g_kernel->modules[j].shutdown_func(&g_kernel->modules[j]);
                    }
                }
                return result;
            }
            module->initialized = true;
        }
        module->active = true;
    }
    
    /* Start message processing thread */
    atomic_store(&g_kernel->should_stop, false);
    if (pthread_create(&g_kernel->message_thread, NULL, message_thread_func, g_kernel) != 0) {
        return CE_ERROR_UNKNOWN;
    }
    
    /* Start main loop thread */
    if (pthread_create(&g_kernel->main_thread, NULL, main_loop_thread_func, g_kernel) != 0) {
        atomic_store(&g_kernel->should_stop, true);
        pthread_join(g_kernel->message_thread, NULL);
        return CE_ERROR_UNKNOWN;
    }
    
    g_kernel->running = true;
    
    /* Wait for duration if specified */
    if (duration_seconds > 0) {
        usleep((useconds_t)(duration_seconds * 1000000));
        ce_kernel_stop();
    }
    
    return CE_SUCCESS;
}

ce_error_t ce_kernel_stop(void) {
    if (!g_kernel || !g_kernel->running) {
        return CE_ERROR_INVALID_STATE;
    }
    
    /* Signal threads to stop */
    atomic_store(&g_kernel->should_stop, true);
    
    /* Wait for threads to finish */
    pthread_join(g_kernel->main_thread, NULL);
    pthread_join(g_kernel->message_thread, NULL);
    
    g_kernel->running = false;
    g_kernel->total_runtime = get_current_time() - g_kernel->start_time;
    
    return CE_SUCCESS;
}

ce_error_t ce_kernel_tick(void) {
    if (!g_kernel) {
        return CE_ERROR_INVALID_STATE;
    }
    
    double current_time = get_current_time();
    
    /* Update working memory */
    if (g_kernel->global_wm) {
        ce_wm_update(g_kernel->global_wm);
    }
    
    /* Process workspace */
    if (g_kernel->global_workspace) {
        ce_workspace_process(g_kernel->global_workspace);
        
        /* Get broadcasted items and send to reasoner */
        const ce_item_list_t *broadcast = ce_workspace_get_broadcast(g_kernel->global_workspace);
        if (broadcast && g_kernel->global_reasoner) {
            ce_item_list_t *reasoning_results = ce_reasoner_process(g_kernel->global_reasoner, broadcast);
            if (reasoning_results) {
                /* Add reasoning results back to working memory */
                for (size_t i = 0; i < reasoning_results->count; i++) {
                    ce_wm_add(g_kernel->global_wm, reasoning_results->items[i]);
                }
                ce_item_list_free(reasoning_results);
            }
        }
    }
    
    /* Update self-model */
    if (g_kernel->global_self_model && g_kernel->global_workspace) {
        const ce_item_list_t *recent_thoughts = ce_wm_get_items(g_kernel->global_wm);
        ce_self_model_update(g_kernel->global_self_model, g_kernel->global_workspace, recent_thoughts);
    }
    
    /* Tick all registered modules */
    for (size_t i = 0; i < g_kernel->module_count; i++) {
        ce_module_t *module = &g_kernel->modules[i];
        if (module->active && module->tick_func) {
            double time_since_last_tick = current_time - module->last_tick;
            if (time_since_last_tick >= module->tick_interval) {
                double tick_start = get_current_time();
                module->tick_func(module, current_time);
                double tick_end = get_current_time();
                
                module->last_tick = current_time;
                module->tick_count++;
                module->total_tick_time += (tick_end - tick_start);
                
                if ((tick_end - tick_start) > module->max_tick_time) {
                    module->max_tick_time = tick_end - tick_start;
                }
            }
        }
    }
    
    return CE_SUCCESS;
}

ce_error_t ce_kernel_broadcast_item(const ce_item_t *item) {
    if (!g_kernel || !item) {
        return CE_ERROR_NULL_POINTER;
    }
    
    ce_message_t *message = ce_message_create(CE_MESSAGE_TYPE_BROADCAST, "kernel", "", (ce_item_t *)item);
    if (!message) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    ce_error_t result = ce_kernel_send_message(message);
    ce_message_free(message);
    
    return result;
}

ce_error_t ce_kernel_send_message(const ce_message_t *message) {
    if (!g_kernel || !message) {
        return CE_ERROR_NULL_POINTER;
    }
    
    return ce_message_queue_enqueue(&g_kernel->message_queue, message);
}

/* ============================================================================
 * Module Management
 * ============================================================================ */

ce_module_t *ce_kernel_register_module(const char *name, ce_module_type_t type,
                                      void *context,
                                      ce_module_init_func_t init_func,
                                      ce_module_tick_func_t tick_func,
                                      ce_module_shutdown_func_t shutdown_func,
                                      ce_module_broadcast_func_t broadcast_func,
                                      ce_module_serialize_func_t serialize_func,
                                      ce_module_deserialize_func_t deserialize_func) {
    if (!g_kernel || !name) {
        return NULL;
    }
    
    if (g_kernel->module_count >= CE_KERNEL_MAX_MODULES) {
        return NULL;
    }
    
    ce_module_t *module = &g_kernel->modules[g_kernel->module_count];
    
    /* Initialize module structure */
    strncpy(module->name, name, sizeof(module->name) - 1);
    module->name[sizeof(module->name) - 1] = '\0';
    module->type = type;
    module->context = context;
    module->init_func = init_func;
    module->tick_func = tick_func;
    module->shutdown_func = shutdown_func;
    module->broadcast_func = broadcast_func;
    module->serialize_func = serialize_func;
    module->deserialize_func = deserialize_func;
    module->initialized = false;
    module->active = false;
    module->last_tick = 0.0;
    module->tick_interval = 0.0; /* No minimum interval by default */
    module->tick_count = 0;
    module->broadcast_count = 0;
    module->total_tick_time = 0.0;
    module->max_tick_time = 0.0;
    
    g_kernel->module_count++;
    
    return module;
}

ce_error_t ce_kernel_unregister_module(ce_module_t *module) {
    if (!g_kernel || !module) {
        return CE_ERROR_NULL_POINTER;
    }
    
    /* Find module in array */
    size_t module_index = SIZE_MAX;
    for (size_t i = 0; i < g_kernel->module_count; i++) {
        if (&g_kernel->modules[i] == module) {
            module_index = i;
            break;
        }
    }
    
    if (module_index == SIZE_MAX) {
        return CE_ERROR_MODULE_NOT_FOUND;
    }
    
    /* Shutdown module if initialized */
    if (module->initialized && module->shutdown_func) {
        module->shutdown_func(module);
    }
    
    /* Shift remaining modules */
    for (size_t i = module_index; i < g_kernel->module_count - 1; i++) {
        g_kernel->modules[i] = g_kernel->modules[i + 1];
    }
    
    g_kernel->module_count--;
    
    return CE_SUCCESS;
}

/* ============================================================================
 * Global Module Accessors
 * ============================================================================ */

ce_working_memory_t *ce_kernel_get_global_wm(void) {
    return g_kernel ? g_kernel->global_wm : NULL;
}

ce_workspace_t *ce_kernel_get_global_workspace(void) {
    return g_kernel ? g_kernel->global_workspace : NULL;
}

ce_long_term_memory_t *ce_kernel_get_global_ltm(void) {
    return g_kernel ? g_kernel->global_ltm : NULL;
}

ce_reasoner_t *ce_kernel_get_global_reasoner(void) {
    return g_kernel ? g_kernel->global_reasoner : NULL;
}

ce_self_model_t *ce_kernel_get_global_self_model(void) {
    return g_kernel ? g_kernel->global_self_model : NULL;
}

ce_kernel_t *ce_kernel_get_instance(void) {
    return g_kernel;
}

ce_module_t *ce_kernel_get_module(const char *name) {
    if (!g_kernel || !name) {
        return NULL;
    }
    
    for (size_t i = 0; i < g_kernel->module_count; i++) {
        if (strcmp(g_kernel->modules[i].name, name) == 0) {
            return &g_kernel->modules[i];
        }
    }
    
    return NULL;
}

ce_module_t *ce_kernel_get_module_by_type(ce_module_type_t type) {
    if (!g_kernel) {
        return NULL;
    }
    
    for (size_t i = 0; i < g_kernel->module_count; i++) {
        if (g_kernel->modules[i].type == type) {
            return &g_kernel->modules[i];
        }
    }
    
    return NULL;
}

/* ============================================================================
 * Statistics and Control
 * ============================================================================ */

ce_error_t ce_kernel_get_stats(struct {
    double total_runtime;
    uint64_t total_ticks;
    double avg_tick_time;
    double max_tick_time;
    size_t active_modules;
    size_t message_queue_size;
} *stats) {
    if (!g_kernel || !stats) {
        return CE_ERROR_NULL_POINTER;
    }
    
    stats->total_runtime = g_kernel->total_runtime;
    stats->total_ticks = g_kernel->total_ticks;
    stats->avg_tick_time = g_kernel->avg_tick_time;
    stats->max_tick_time = g_kernel->max_tick_time;
    stats->active_modules = 0;
    stats->message_queue_size = ce_message_queue_size(&g_kernel->message_queue);
    
    for (size_t i = 0; i < g_kernel->module_count; i++) {
        if (g_kernel->modules[i].active) {
            stats->active_modules++;
        }
    }
    
    return CE_SUCCESS;
}

ce_error_t ce_kernel_pause(void) {
    if (!g_kernel) {
        return CE_ERROR_INVALID_STATE;
    }
    
    g_kernel->paused = true;
    return CE_SUCCESS;
}

ce_error_t ce_kernel_resume(void) {
    if (!g_kernel) {
        return CE_ERROR_INVALID_STATE;
    }
    
    g_kernel->paused = false;
    return CE_SUCCESS;
}
