/**
 * Consciousness Emulator - Working Memory Module
 * 
 * Implements a fixed-capacity working memory with saliency-based attention
 * and decay mechanisms. This is the short-term cognitive buffer.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#ifndef CE_WORKING_MEMORY_H
#define CE_WORKING_MEMORY_H

#include "../../include/consciousness.h"
#include <stdbool.h>

/* ============================================================================
 * Working Memory Configuration
 * ============================================================================ */

#define CE_WM_DEFAULT_CAPACITY 16
#define CE_WM_MAX_CAPACITY 128
#define CE_WM_DEFAULT_DECAY_RATE 0.95f
#define CE_WM_DEFAULT_SALIENCY_THRESHOLD 0.1f
#define CE_WM_DEFAULT_ACCESS_BOOST 0.1f

/* ============================================================================
 * Working Memory Slot Structure
 * ============================================================================ */

typedef struct {
    ce_item_t *item;                /* The cognitive item */
    double created_at;              /* Creation timestamp */
    double last_accessed;           /* Last access timestamp */
    float base_saliency;            /* Base saliency score */
    float current_saliency;         /* Current saliency (with decay) */
    float access_count;             /* Number of times accessed */
    bool is_active;                 /* Whether slot is currently active */
} ce_wm_slot_t;

/* ============================================================================
 * Working Memory Structure
 * ============================================================================ */

typedef struct ce_working_memory {
    ce_wm_slot_t *slots;            /* Array of memory slots */
    size_t capacity;                /* Maximum number of slots */
    size_t count;                   /* Current number of active slots */
    
    /* Configuration */
    float decay_rate;               /* Saliency decay rate per tick */
    float saliency_threshold;       /* Minimum saliency to remain active */
    float access_boost;             /* Saliency boost per access */
    
    /* Statistics */
    uint64_t total_items_added;
    uint64_t total_items_removed;
    uint64_t total_accesses;
    double total_saliency;
    double avg_saliency;
    
    /* Synchronization */
    pthread_mutex_t mutex;
    
    /* Callbacks */
    void (*on_item_added)(ce_item_t *item, void *user_data);
    void (*on_item_removed)(ce_item_t *item, void *user_data);
    void (*on_item_accessed)(ce_item_t *item, void *user_data);
    void *callback_user_data;
} ce_working_memory_t;

/* ============================================================================
 * Working Memory API
 * ============================================================================ */

/**
 * Create a new working memory instance
 * @param capacity Maximum number of items
 * @return Working memory instance or NULL on error
 */
ce_working_memory_t *ce_wm_create(size_t capacity);

/**
 * Create working memory with custom configuration
 * @param capacity Maximum number of items
 * @param decay_rate Saliency decay rate
 * @param saliency_threshold Minimum saliency threshold
 * @param access_boost Saliency boost per access
 * @return Working memory instance or NULL on error
 */
ce_working_memory_t *ce_wm_create_with_config(size_t capacity, float decay_rate,
                                             float saliency_threshold, float access_boost);

/**
 * Free working memory instance
 * @param wm Working memory instance
 */
void ce_wm_free(ce_working_memory_t *wm);

/**
 * Add an item to working memory
 * @param wm Working memory instance
 * @param item Item to add
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_add(ce_working_memory_t *wm, ce_item_t *item);

/**
 * Remove an item from working memory
 * @param wm Working memory instance
 * @param item_id ID of item to remove
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_remove(ce_working_memory_t *wm, uint64_t item_id);

/**
 * Access an item (boost its saliency)
 * @param wm Working memory instance
 * @param item_id ID of item to access
 * @return Item or NULL if not found
 */
ce_item_t *ce_wm_access(ce_working_memory_t *wm, uint64_t item_id);

/**
 * Find an item by ID
 * @param wm Working memory instance
 * @param item_id ID to search for
 * @return Item or NULL if not found
 */
ce_item_t *ce_wm_find(ce_working_memory_t *wm, uint64_t item_id);

/**
 * Get all items in working memory
 * @param wm Working memory instance
 * @return Item list (do not free, owned by WM)
 */
const ce_item_list_t *ce_wm_get_items(const ce_working_memory_t *wm);

/**
 * Get top-k items by saliency
 * @param wm Working memory instance
 * @param k Number of items to return
 * @param result Output array (must be pre-allocated)
 * @return Number of items returned
 */
size_t ce_wm_get_topk(const ce_working_memory_t *wm, size_t k, ce_item_t **result);

/**
 * Get items above saliency threshold
 * @param wm Working memory instance
 * @param threshold Saliency threshold
 * @param result Output array (must be pre-allocated)
 * @param max_results Maximum number of results
 * @return Number of items returned
 */
size_t ce_wm_get_above_threshold(const ce_working_memory_t *wm, float threshold,
                                 ce_item_t **result, size_t max_results);

/**
 * Update working memory (decay, consolidation)
 * @param wm Working memory instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_update(ce_working_memory_t *wm);

/**
 * Force decay of all items
 * @param wm Working memory instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_decay(ce_working_memory_t *wm);

/**
 * Consolidate similar items
 * @param wm Working memory instance
 * @param similarity_threshold Similarity threshold for consolidation
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_consolidate(ce_working_memory_t *wm, float similarity_threshold);

/**
 * Clear all items from working memory
 * @param wm Working memory instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_clear(ce_working_memory_t *wm);

/**
 * Get working memory statistics
 * @param wm Working memory instance
 * @param stats Output statistics structure
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_get_stats(const ce_working_memory_t *wm, struct {
    size_t capacity;
    size_t count;
    size_t active_slots;
    double total_saliency;
    double avg_saliency;
    double max_saliency;
    double min_saliency;
    uint64_t total_items_added;
    uint64_t total_items_removed;
    uint64_t total_accesses;
} *stats);

/**
 * Set callback functions
 * @param wm Working memory instance
 * @param on_item_added Callback for item addition
 * @param on_item_removed Callback for item removal
 * @param on_item_accessed Callback for item access
 * @param user_data User data passed to callbacks
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_set_callbacks(ce_working_memory_t *wm,
                               void (*on_item_added)(ce_item_t *item, void *user_data),
                               void (*on_item_removed)(ce_item_t *item, void *user_data),
                               void (*on_item_accessed)(ce_item_t *item, void *user_data),
                               void *user_data);

/**
 * Serialize working memory to buffer
 * @param wm Working memory instance
 * @param buffer Output buffer (caller must free)
 * @param size Output buffer size
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_serialize(const ce_working_memory_t *wm, char **buffer, size_t *size);

/**
 * Deserialize working memory from buffer
 * @param wm Working memory instance
 * @param buffer Input buffer
 * @param size Buffer size
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_deserialize(ce_working_memory_t *wm, const char *buffer, size_t size);

/* ============================================================================
 * Internal Functions
 * ============================================================================ */

/**
 * Find the best slot for a new item (lowest saliency or empty)
 * @param wm Working memory instance
 * @return Slot index or SIZE_MAX if no space
 */
size_t ce_wm_find_best_slot(const ce_working_memory_t *wm);

/**
 * Update saliency of a slot
 * @param slot Memory slot
 * @param decay_rate Decay rate
 * @param current_time Current timestamp
 */
void ce_wm_update_slot_saliency(ce_wm_slot_t *slot, float decay_rate, double current_time);

/**
 * Remove items below saliency threshold
 * @param wm Working memory instance
 * @return Number of items removed
 */
size_t ce_wm_remove_low_saliency(ce_working_memory_t *wm);

/**
 * Compute similarity between two items
 * @param item1 First item
 * @param item2 Second item
 * @return Similarity score [0.0, 1.0]
 */
float ce_wm_compute_similarity(const ce_item_t *item1, const ce_item_t *item2);

/**
 * Merge two similar items
 * @param wm Working memory instance
 * @param slot1 First slot index
 * @param slot2 Second slot index
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_wm_merge_slots(ce_working_memory_t *wm, size_t slot1, size_t slot2);

#endif /* CE_WORKING_MEMORY_H */
