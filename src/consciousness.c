/**
 * Consciousness Emulator - Main Implementation
 * 
 * This file implements the main API functions that coordinate all
 * cognitive modules and provide the unified interface.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "../include/consciousness.h"
#include "kernel/kernel.h"
#include "wm/working_memory.h"
#include "utils/math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

/* ============================================================================
 * Global State
 * ============================================================================ */

static bool g_initialized = false;
static uint64_t g_item_id_counter = 1;
static uint64_t g_random_seed = 1;

/* ============================================================================
 * Core System Functions
 * ============================================================================ */

ce_error_t ce_init(double tick_hz) {
    if (g_initialized) {
        return CE_ERROR_INVALID_STATE;
    }
    
    /* Initialize random number generator */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    g_random_seed = tv.tv_sec ^ tv.tv_usec;
    ce_random_init((uint32_t)g_random_seed);
    
    /* Initialize kernel */
    ce_error_t result = ce_kernel_init(tick_hz);
    if (result != CE_SUCCESS) {
        return result;
    }
    
    g_initialized = true;
    return CE_SUCCESS;
}

ce_error_t ce_shutdown(void) {
    if (!g_initialized) {
        return CE_ERROR_INVALID_STATE;
    }
    
    /* Shutdown kernel */
    ce_error_t result = ce_kernel_shutdown();
    if (result != CE_SUCCESS) {
        return result;
    }
    
    g_initialized = false;
    return CE_SUCCESS;
}

ce_error_t ce_run(double duration_seconds) {
    if (!g_initialized) {
        return CE_ERROR_INVALID_STATE;
    }
    
    return ce_kernel_start(duration_seconds);
}

ce_error_t ce_tick(void) {
    if (!g_initialized) {
        return CE_ERROR_INVALID_STATE;
    }
    
    return ce_kernel_tick();
}

double ce_get_timestamp(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

/* ============================================================================
 * Item Management
 * ============================================================================ */

ce_item_t *ce_item_create(ce_item_type_t type, const char *content, float confidence) {
    if (!content) {
        return NULL;
    }
    
    ce_item_t *item = calloc(1, sizeof(ce_item_t));
    if (!item) {
        return NULL;
    }
    
    item->id = g_item_id_counter++;
    item->type = type;
    item->timestamp = ce_get_timestamp();
    item->last_accessed = item->timestamp;
    item->confidence = confidence;
    item->saliency = confidence; /* Initial saliency based on confidence */
    item->embedding = NULL;
    item->embedding_dim = 0;
    item->provenance = NULL;
    item->next = NULL;
    
    /* Copy content */
    item->content = malloc(strlen(content) + 1);
    if (!item->content) {
        free(item);
        return NULL;
    }
    strcpy(item->content, content);
    
    /* Initialize metadata */
    item->metadata = NULL;
    item->payload = NULL;
    
    return item;
}

ce_item_t *ce_item_create_with_embedding(ce_item_type_t type, const char *content,
                                        const float *embedding, size_t embedding_dim,
                                        float confidence) {
    ce_item_t *item = ce_item_create(type, content, confidence);
    if (!item) {
        return NULL;
    }
    
    if (embedding && embedding_dim > 0) {
        item->embedding = malloc(embedding_dim * sizeof(float));
        if (!item->embedding) {
            ce_item_free(item);
            return NULL;
        }
        
        memcpy(item->embedding, embedding, embedding_dim * sizeof(float));
        item->embedding_dim = embedding_dim;
    }
    
    return item;
}

ce_item_t *ce_item_clone(const ce_item_t *item) {
    if (!item) {
        return NULL;
    }
    
    ce_item_t *clone = malloc(sizeof(ce_item_t));
    if (!clone) {
        return NULL;
    }
    
    /* Copy basic fields */
    *clone = *item;
    clone->id = g_item_id_counter++;
    clone->timestamp = ce_get_timestamp();
    clone->last_accessed = clone->timestamp;
    
    /* Copy content */
    if (item->content) {
        clone->content = malloc(strlen(item->content) + 1);
        if (!clone->content) {
            free(clone);
            return NULL;
        }
        strcpy(clone->content, item->content);
    } else {
        clone->content = NULL;
    }
    
    /* Copy metadata */
    if (item->metadata) {
        clone->metadata = malloc(strlen(item->metadata) + 1);
        if (!clone->metadata) {
            free(clone->content);
            free(clone);
            return NULL;
        }
        strcpy(clone->metadata, item->metadata);
    } else {
        clone->metadata = NULL;
    }
    
    /* Copy embedding */
    if (item->embedding && item->embedding_dim > 0) {
        clone->embedding = malloc(item->embedding_dim * sizeof(float));
        if (!clone->embedding) {
            free(clone->content);
            free(clone->metadata);
            free(clone);
            return NULL;
        }
        memcpy(clone->embedding, item->embedding, item->embedding_dim * sizeof(float));
    } else {
        clone->embedding = NULL;
    }
    
    /* Don't copy provenance or next - these are references */
    clone->provenance = NULL;
    clone->next = NULL;
    clone->payload = NULL; /* Payload is not cloned */
    
    return clone;
}

void ce_item_update_saliency(ce_item_t *item, float saliency) {
    if (item) {
        item->saliency = saliency;
        item->last_accessed = ce_get_timestamp();
    }
}

void ce_item_update_confidence(ce_item_t *item, float confidence) {
    if (item) {
        item->confidence = confidence;
        item->last_accessed = ce_get_timestamp();
    }
}

void ce_item_free(ce_item_t *item) {
    if (!item) {
        return;
    }
    
    if (item->content) {
        free(item->content);
    }
    
    if (item->metadata) {
        free(item->metadata);
    }
    
    if (item->embedding) {
        free(item->embedding);
    }
    
    if (item->payload) {
        free(item->payload);
    }
    
    free(item);
}

/* ============================================================================
 * Item List Management
 * ============================================================================ */

ce_item_list_t *ce_item_list_create(size_t initial_capacity) {
    ce_item_list_t *list = malloc(sizeof(ce_item_list_t));
    if (!list) {
        return NULL;
    }
    
    list->items = calloc(initial_capacity, sizeof(ce_item_t *));
    if (!list->items) {
        free(list);
        return NULL;
    }
    
    list->count = 0;
    list->capacity = initial_capacity;
    list->total_saliency = 0.0f;
    
    return list;
}

ce_error_t ce_item_list_add(ce_item_list_t *list, ce_item_t *item) {
    if (!list || !item) {
        return CE_ERROR_NULL_POINTER;
    }
    
    if (list->count >= list->capacity) {
        /* Resize array */
        size_t new_capacity = list->capacity * 2;
        ce_item_t **new_items = realloc(list->items, new_capacity * sizeof(ce_item_t *));
        if (!new_items) {
            return CE_ERROR_OUT_OF_MEMORY;
        }
        
        list->items = new_items;
        list->capacity = new_capacity;
    }
    
    list->items[list->count] = item;
    list->count++;
    list->total_saliency += item->saliency;
    
    return CE_SUCCESS;
}

ce_error_t ce_item_list_remove(ce_item_list_t *list, ce_item_t *item) {
    if (!list || !item) {
        return CE_ERROR_NULL_POINTER;
    }
    
    for (size_t i = 0; i < list->count; i++) {
        if (list->items[i] == item) {
            /* Shift remaining items */
            for (size_t j = i; j < list->count - 1; j++) {
                list->items[j] = list->items[j + 1];
            }
            
            list->count--;
            list->total_saliency -= item->saliency;
            return CE_SUCCESS;
        }
    }
    
    return CE_ERROR_UNKNOWN; /* Item not found */
}

size_t ce_item_list_topk(const ce_item_list_t *list, size_t k, ce_item_t **result) {
    if (!list || !result || k == 0) {
        return 0;
    }
    
    /* Create array of indices with saliencies */
    struct {
        size_t index;
        float saliency;
    } *saliency_array = malloc(list->count * sizeof(*saliency_array));
    
    if (!saliency_array) {
        return 0;
    }
    
    for (size_t i = 0; i < list->count; i++) {
        saliency_array[i].index = i;
        saliency_array[i].saliency = list->items[i]->saliency;
    }
    
    /* Sort by saliency (simple bubble sort) */
    for (size_t i = 0; i < list->count - 1; i++) {
        for (size_t j = 0; j < list->count - i - 1; j++) {
            if (saliency_array[j].saliency < saliency_array[j + 1].saliency) {
                struct { size_t index; float saliency; } temp = saliency_array[j];
                saliency_array[j] = saliency_array[j + 1];
                saliency_array[j + 1] = temp;
            }
        }
    }
    
    /* Return top k items */
    size_t result_count = (k < list->count) ? k : list->count;
    for (size_t i = 0; i < result_count; i++) {
        result[i] = list->items[saliency_array[i].index];
    }
    
    free(saliency_array);
    return result_count;
}

void ce_item_list_free(ce_item_list_t *list) {
    if (!list) {
        return;
    }
    
    if (list->items) {
        free(list->items);
    }
    
    free(list);
}

/* ============================================================================
 * Working Memory Interface
 * ============================================================================ */

ce_working_memory_t *ce_wm_create(size_t capacity) {
    return ce_wm_create(capacity);
}

ce_error_t ce_wm_add(ce_working_memory_t *wm, ce_item_t *item) {
    return ce_wm_add(wm, item);
}

const ce_item_list_t *ce_wm_get_items(const ce_working_memory_t *wm) {
    return ce_wm_get_items(wm);
}

ce_error_t ce_wm_update(ce_working_memory_t *wm) {
    return ce_wm_update(wm);
}

void ce_wm_free(ce_working_memory_t *wm) {
    ce_wm_free(wm);
}

/* ============================================================================
 * Global Workspace Interface
 * ============================================================================ */

ce_workspace_t *ce_workspace_create(ce_working_memory_t *wm, float threshold) {
    /* This will be implemented in the workspace module */
    return NULL;
}

ce_error_t ce_workspace_process(ce_workspace_t *workspace) {
    /* This will be implemented in the workspace module */
    return CE_ERROR_UNKNOWN;
}

const ce_item_list_t *ce_workspace_get_broadcast(const ce_workspace_t *workspace) {
    /* This will be implemented in the workspace module */
    return NULL;
}

void ce_workspace_free(ce_workspace_t *workspace) {
    /* This will be implemented in the workspace module */
}

/* ============================================================================
 * Long-Term Memory Interface
 * ============================================================================ */

ce_long_term_memory_t *ce_ltm_create(size_t embedding_dim, size_t max_episodes) {
    /* This will be implemented in the LTM module */
    return NULL;
}

ce_error_t ce_ltm_store_episode(ce_long_term_memory_t *ltm, const ce_item_t *item, const char *context) {
    /* This will be implemented in the LTM module */
    return CE_ERROR_UNKNOWN;
}

size_t ce_ltm_search(ce_long_term_memory_t *ltm, const float *query_embedding, 
                     size_t k, ce_item_t **results) {
    /* This will be implemented in the LTM module */
    return 0;
}

ce_error_t ce_ltm_consolidate(ce_long_term_memory_t *ltm) {
    /* This will be implemented in the LTM module */
    return CE_ERROR_UNKNOWN;
}

void ce_ltm_free(ce_long_term_memory_t *ltm) {
    /* This will be implemented in the LTM module */
}

/* ============================================================================
 * Reasoning Engine Interface
 * ============================================================================ */

ce_reasoner_t *ce_reasoner_create(void) {
    /* This will be implemented in the reasoner module */
    return NULL;
}

ce_item_list_t *ce_reasoner_process(ce_reasoner_t *reasoner, const ce_item_list_t *broadcast_items) {
    /* This will be implemented in the reasoner module */
    return NULL;
}

ce_error_t ce_reasoner_add_rule(ce_reasoner_t *reasoner, const char *rule_name,
                               bool (*condition)(const ce_item_t *),
                               ce_item_t *(*action)(const ce_item_t *)) {
    /* This will be implemented in the reasoner module */
    return CE_ERROR_UNKNOWN;
}

void ce_reasoner_free(ce_reasoner_t *reasoner) {
    /* This will be implemented in the reasoner module */
}

/* ============================================================================
 * Self-Model Interface
 * ============================================================================ */

ce_self_model_t *ce_self_model_create(void) {
    /* This will be implemented in the self-model module */
    return NULL;
}

ce_error_t ce_self_model_update(ce_self_model_t *self_model, 
                               const ce_workspace_t *workspace,
                               const ce_item_list_t *recent_thoughts) {
    /* This will be implemented in the self-model module */
    return CE_ERROR_UNKNOWN;
}

char *ce_self_model_get_summary(const ce_self_model_t *self_model) {
    /* This will be implemented in the self-model module */
    return NULL;
}

char *ce_self_model_explain_decision(const ce_self_model_t *self_model, const ce_item_t *item) {
    /* This will be implemented in the self-model module */
    return NULL;
}

void ce_self_model_free(ce_self_model_t *self_model) {
    /* This will be implemented in the self-model module */
}

/* ============================================================================
 * IO and Control Interface
 * ============================================================================ */

static ce_input_handler_t g_input_handler = NULL;
static void *g_input_user_data = NULL;
static ce_output_handler_t g_output_handler = NULL;
static void *g_output_user_data = NULL;

ce_error_t ce_io_register_input_handler(ce_input_handler_t handler, void *user_data) {
    g_input_handler = handler;
    g_input_user_data = user_data;
    return CE_SUCCESS;
}

ce_error_t ce_io_register_output_handler(ce_output_handler_t handler, void *user_data) {
    g_output_handler = handler;
    g_output_user_data = user_data;
    return CE_SUCCESS;
}

ce_error_t ce_io_send_input(const char *input) {
    if (!input || !g_input_handler) {
        return CE_ERROR_NULL_POINTER;
    }
    
    ce_item_t *item = g_input_handler(input, g_input_user_data);
    if (item) {
        /* Add to global working memory */
        ce_working_memory_t *wm = ce_kernel_get_global_wm();
        if (wm) {
            ce_wm_add(wm, item);
        }
        ce_item_free(item);
    }
    
    return CE_SUCCESS;
}

size_t ce_io_get_output(char *output, size_t max_length) {
    if (!output || max_length == 0) {
        return 0;
    }
    
    /* Get self-model summary */
    ce_self_model_t *self_model = ce_kernel_get_global_self_model();
    if (self_model) {
        char *summary = ce_self_model_get_summary(self_model);
        if (summary) {
            size_t len = strlen(summary);
            if (len >= max_length) {
                len = max_length - 1;
            }
            strncpy(output, summary, len);
            output[len] = '\0';
            free(summary);
            return len;
        }
    }
    
    /* Fallback to simple message */
    strncpy(output, "Consciousness Emulator is running...", max_length - 1);
    output[max_length - 1] = '\0';
    return strlen(output);
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

const char *ce_error_string(ce_error_t error) {
    switch (error) {
        case CE_SUCCESS: return "Success";
        case CE_ERROR_NULL_POINTER: return "Null pointer error";
        case CE_ERROR_INVALID_PARAM: return "Invalid parameter";
        case CE_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case CE_ERROR_MODULE_NOT_FOUND: return "Module not found";
        case CE_ERROR_INVALID_STATE: return "Invalid state";
        case CE_ERROR_SERIALIZATION: return "Serialization error";
        case CE_ERROR_DESERIALIZATION: return "Deserialization error";
        case CE_ERROR_IO: return "I/O error";
        case CE_ERROR_MATH: return "Mathematical error";
        case CE_ERROR_UNKNOWN: return "Unknown error";
        default: return "Invalid error code";
    }
}

const char *ce_get_version(void) {
    static char version[32];
    snprintf(version, sizeof(version), "%d.%d.%d", 
             CE_VERSION_MAJOR, CE_VERSION_MINOR, CE_VERSION_PATCH);
    return version;
}


void ce_generate_random_embedding(float *embedding, size_t dim, uint32_t seed) {
    if (!embedding || dim == 0) {
        return;
    }
    
    uint32_t old_seed = g_random_seed;
    ce_random_init(seed);
    
    for (size_t i = 0; i < dim; i++) {
        embedding[i] = ce_random_float_range(-1.0f, 1.0f);
    }
    
    ce_random_init(old_seed);
}

float ce_cosine_similarity(const float *a, const float *b, size_t dim) {
    return ce_cosine_similarity(a, b, dim);
}

float ce_l2_distance(const float *a, const float *b, size_t dim) {
    return ce_l2_distance(a, b, dim);
}
