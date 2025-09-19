/**
 * Consciousness Emulator - Working Memory Implementation
 * 
 * A sophisticated working memory system with saliency-based attention,
 * decay mechanisms, and consolidation capabilities.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "working_memory.h"
#include "../utils/math_utils.h"
#include "../utils/arena.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * Compute similarity between two items using embedding cosine similarity
 */
static float compute_item_similarity(const ce_item_t *item1, const ce_item_t *item2) {
    if (!item1 || !item2) {
        return 0.0f;
    }
    
    /* If both have embeddings, use cosine similarity */
    if (item1->embedding && item2->embedding && 
        item1->embedding_dim == item2->embedding_dim) {
        return ce_cosine_similarity(item1->embedding, item2->embedding, item1->embedding_dim);
    }
    
    /* Fallback to content similarity */
    if (item1->content && item2->content) {
        if (strcmp(item1->content, item2->content) == 0) {
            return 1.0f;
        }
        
        /* Simple string similarity based on common words */
        char *content1 = strdup(item1->content);
        char *content2 = strdup(item2->content);
        
        if (!content1 || !content2) {
            free(content1);
            free(content2);
            return 0.0f;
        }
        
        /* Convert to lowercase */
        for (char *p = content1; *p; p++) *p = tolower(*p);
        for (char *p = content2; *p; p++) *p = tolower(*p);
        
        /* Count common words */
        int common_words = 0;
        int total_words = 0;
        
        char *word1 = strtok(content1, " \t\n\r");
        while (word1) {
            total_words++;
            char *word2 = strtok(content2, " \t\n\r");
            while (word2) {
                if (strcmp(word1, word2) == 0) {
                    common_words++;
                    break;
                }
                word2 = strtok(NULL, " \t\n\r");
            }
            word1 = strtok(NULL, " \t\n\r");
        }
        
        free(content1);
        free(content2);
        
        return total_words > 0 ? (float)common_words / total_words : 0.0f;
    }
    
    return 0.0f;
}

/**
 * Update slot saliency with decay
 */
static void update_slot_saliency(ce_wm_slot_t *slot, float decay_rate, double current_time) {
    if (!slot || !slot->item) {
        return;
    }
    
    /* Apply decay based on time since last access */
    double time_since_access = current_time - slot->last_accessed;
    float time_decay = (float)exp(-decay_rate * time_since_access);
    
    /* Update current saliency */
    slot->current_saliency = slot->base_saliency * time_decay;
    
    /* Ensure saliency doesn't go below zero */
    if (slot->current_saliency < 0.0f) {
        slot->current_saliency = 0.0f;
    }
    
    /* Update item saliency */
    slot->item->saliency = slot->current_saliency;
}

/**
 * Find the best slot for a new item
 */
static size_t find_best_slot(const ce_working_memory_t *wm) {
    size_t best_slot = SIZE_MAX;
    float lowest_saliency = 1.0f;
    
    for (size_t i = 0; i < wm->capacity; i++) {
        const ce_wm_slot_t *slot = &wm->slots[i];
        
        if (!slot->is_active) {
            /* Empty slot is always best */
            return i;
        }
        
        if (slot->current_saliency < lowest_saliency) {
            lowest_saliency = slot->current_saliency;
            best_slot = i;
        }
    }
    
    return best_slot;
}

/**
 * Remove items below saliency threshold
 */
static size_t remove_low_saliency_items(ce_working_memory_t *wm) {
    size_t removed_count = 0;
    
    for (size_t i = 0; i < wm->capacity; i++) {
        ce_wm_slot_t *slot = &wm->slots[i];
        
        if (slot->is_active && slot->current_saliency < wm->saliency_threshold) {
            /* Call removal callback */
            if (wm->on_item_removed && slot->item) {
                wm->on_item_removed(slot->item, wm->callback_user_data);
            }
            
            /* Free the item */
            if (slot->item) {
                ce_item_free(slot->item);
            }
            
            /* Clear the slot */
            memset(slot, 0, sizeof(ce_wm_slot_t));
            slot->is_active = false;
            
            wm->count--;
            wm->total_items_removed++;
            removed_count++;
        }
    }
    
    return removed_count;
}

/* ============================================================================
 * Public API Implementation
 * ============================================================================ */

ce_working_memory_t *ce_wm_create(size_t capacity) {
    return ce_wm_create_with_config(capacity, CE_WM_DEFAULT_DECAY_RATE,
                                   CE_WM_DEFAULT_SALIENCY_THRESHOLD,
                                   CE_WM_DEFAULT_ACCESS_BOOST);
}

ce_working_memory_t *ce_wm_create_with_config(size_t capacity, float decay_rate,
                                             float saliency_threshold, float access_boost) {
    if (capacity == 0 || capacity > CE_WM_MAX_CAPACITY) {
        return NULL;
    }
    
    ce_working_memory_t *wm = calloc(1, sizeof(ce_working_memory_t));
    if (!wm) {
        return NULL;
    }
    
    wm->slots = calloc(capacity, sizeof(ce_wm_slot_t));
    if (!wm->slots) {
        free(wm);
        return NULL;
    }
    
    wm->capacity = capacity;
    wm->count = 0;
    wm->decay_rate = decay_rate;
    wm->saliency_threshold = saliency_threshold;
    wm->access_boost = access_boost;
    
    /* Initialize statistics */
    wm->total_items_added = 0;
    wm->total_items_removed = 0;
    wm->total_accesses = 0;
    wm->total_saliency = 0.0;
    wm->avg_saliency = 0.0;
    
    /* Initialize synchronization */
    if (pthread_mutex_init(&wm->mutex, NULL) != 0) {
        free(wm->slots);
        free(wm);
        return NULL;
    }
    
    /* Initialize callbacks */
    wm->on_item_added = NULL;
    wm->on_item_removed = NULL;
    wm->on_item_accessed = NULL;
    wm->callback_user_data = NULL;
    
    return wm;
}

void ce_wm_free(ce_working_memory_t *wm) {
    if (!wm) {
        return;
    }
    
    /* Free all items */
    for (size_t i = 0; i < wm->capacity; i++) {
        if (wm->slots[i].item) {
            ce_item_free(wm->slots[i].item);
        }
    }
    
    /* Destroy synchronization */
    pthread_mutex_destroy(&wm->mutex);
    
    /* Free memory */
    free(wm->slots);
    free(wm);
}

ce_error_t ce_wm_add(ce_working_memory_t *wm, ce_item_t *item) {
    if (!wm || !item) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&wm->mutex);
    
    /* Find best slot for the item */
    size_t slot_index = find_best_slot(wm);
    if (slot_index == SIZE_MAX) {
        pthread_mutex_unlock(&wm->mutex);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    ce_wm_slot_t *slot = &wm->slots[slot_index];
    
    /* If slot is active, remove the old item */
    if (slot->is_active && slot->item) {
        if (wm->on_item_removed) {
            wm->on_item_removed(slot->item, wm->callback_user_data);
        }
        ce_item_free(slot->item);
        wm->count--;
        wm->total_items_removed++;
    }
    
    /* Add new item */
    slot->item = ce_item_clone(item);
    if (!slot->item) {
        pthread_mutex_unlock(&wm->mutex);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    slot->created_at = ce_get_timestamp();
    slot->last_accessed = slot->created_at;
    slot->base_saliency = item->saliency;
    slot->current_saliency = item->saliency;
    slot->access_count = 0.0f;
    slot->is_active = true;
    
    wm->count++;
    wm->total_items_added++;
    wm->total_saliency += item->saliency;
    wm->avg_saliency = wm->total_saliency / wm->count;
    
    /* Call addition callback */
    if (wm->on_item_added) {
        wm->on_item_added(slot->item, wm->callback_user_data);
    }
    
    pthread_mutex_unlock(&wm->mutex);
    
    return CE_SUCCESS;
}

ce_error_t ce_wm_remove(ce_working_memory_t *wm, uint64_t item_id) {
    if (!wm) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&wm->mutex);
    
    for (size_t i = 0; i < wm->capacity; i++) {
        ce_wm_slot_t *slot = &wm->slots[i];
        
        if (slot->is_active && slot->item && slot->item->id == item_id) {
            /* Call removal callback */
            if (wm->on_item_removed) {
                wm->on_item_removed(slot->item, wm->callback_user_data);
            }
            
            /* Free the item */
            ce_item_free(slot->item);
            
            /* Clear the slot */
            memset(slot, 0, sizeof(ce_wm_slot_t));
            slot->is_active = false;
            
            wm->count--;
            wm->total_items_removed++;
            
            pthread_mutex_unlock(&wm->mutex);
            return CE_SUCCESS;
        }
    }
    
    pthread_mutex_unlock(&wm->mutex);
    return CE_ERROR_UNKNOWN; /* Item not found */
}

ce_item_t *ce_wm_access(ce_working_memory_t *wm, uint64_t item_id) {
    if (!wm) {
        return NULL;
    }
    
    pthread_mutex_lock(&wm->mutex);
    
    for (size_t i = 0; i < wm->capacity; i++) {
        ce_wm_slot_t *slot = &wm->slots[i];
        
        if (slot->is_active && slot->item && slot->item->id == item_id) {
            /* Update access statistics */
            slot->last_accessed = ce_get_timestamp();
            slot->access_count += 1.0f;
            
            /* Boost saliency */
            slot->base_saliency += wm->access_boost;
            if (slot->base_saliency > 1.0f) {
                slot->base_saliency = 1.0f;
            }
            slot->current_saliency = slot->base_saliency;
            
            /* Update item */
            slot->item->last_accessed = slot->last_accessed;
            slot->item->saliency = slot->current_saliency;
            
            wm->total_accesses++;
            
            /* Call access callback */
            if (wm->on_item_accessed) {
                wm->on_item_accessed(slot->item, wm->callback_user_data);
            }
            
            pthread_mutex_unlock(&wm->mutex);
            return slot->item;
        }
    }
    
    pthread_mutex_unlock(&wm->mutex);
    return NULL;
}

ce_item_t *ce_wm_find(ce_working_memory_t *wm, uint64_t item_id) {
    if (!wm) {
        return NULL;
    }
    
    pthread_mutex_lock(&wm->mutex);
    
    for (size_t i = 0; i < wm->capacity; i++) {
        ce_wm_slot_t *slot = &wm->slots[i];
        
        if (slot->is_active && slot->item && slot->item->id == item_id) {
            pthread_mutex_unlock(&wm->mutex);
            return slot->item;
        }
    }
    
    pthread_mutex_unlock(&wm->mutex);
    return NULL;
}

const ce_item_list_t *ce_wm_get_items(const ce_working_memory_t *wm) {
    if (!wm) {
        return NULL;
    }
    
    /* This is a simplified implementation - in a real system,
     * you'd want to maintain a separate item list that gets updated
     * when items are added/removed for better performance */
    static ce_item_list_t item_list;
    static ce_item_t *item_ptrs[CE_WM_MAX_CAPACITY];
    
    pthread_mutex_lock((pthread_mutex_t *)&wm->mutex);
    
    item_list.items = item_ptrs;
    item_list.count = 0;
    item_list.capacity = CE_WM_MAX_CAPACITY;
    item_list.total_saliency = 0.0f;
    
    for (size_t i = 0; i < wm->capacity; i++) {
        const ce_wm_slot_t *slot = &wm->slots[i];
        
        if (slot->is_active && slot->item) {
            item_list.items[item_list.count] = slot->item;
            item_list.total_saliency += slot->current_saliency;
            item_list.count++;
        }
    }
    
    pthread_mutex_unlock((pthread_mutex_t *)&wm->mutex);
    
    return &item_list;
}

size_t ce_wm_get_topk(const ce_working_memory_t *wm, size_t k, ce_item_t **result) {
    if (!wm || !result || k == 0) {
        return 0;
    }
    
    pthread_mutex_lock((pthread_mutex_t *)&wm->mutex);
    
    /* Create array of slot indices with their saliencies */
    struct {
        size_t index;
        float saliency;
    } *saliency_array = malloc(wm->capacity * sizeof(*saliency_array));
    
    if (!saliency_array) {
        pthread_mutex_unlock((pthread_mutex_t *)&wm->mutex);
        return 0;
    }
    
    size_t active_count = 0;
    for (size_t i = 0; i < wm->capacity; i++) {
        const ce_wm_slot_t *slot = &wm->slots[i];
        
        if (slot->is_active && slot->item) {
            saliency_array[active_count].index = i;
            saliency_array[active_count].saliency = slot->current_saliency;
            active_count++;
        }
    }
    
    /* Sort by saliency (simple bubble sort for small arrays) */
    for (size_t i = 0; i < active_count - 1; i++) {
        for (size_t j = 0; j < active_count - i - 1; j++) {
            if (saliency_array[j].saliency < saliency_array[j + 1].saliency) {
                struct { size_t index; float saliency; } temp = saliency_array[j];
                saliency_array[j] = saliency_array[j + 1];
                saliency_array[j + 1] = temp;
            }
        }
    }
    
    /* Return top k items */
    size_t result_count = (k < active_count) ? k : active_count;
    for (size_t i = 0; i < result_count; i++) {
        result[i] = wm->slots[saliency_array[i].index].item;
    }
    
    free(saliency_array);
    pthread_mutex_unlock((pthread_mutex_t *)&wm->mutex);
    
    return result_count;
}

size_t ce_wm_get_above_threshold(const ce_working_memory_t *wm, float threshold,
                                 ce_item_t **result, size_t max_results) {
    if (!wm || !result || max_results == 0) {
        return 0;
    }
    
    pthread_mutex_lock((pthread_mutex_t *)&wm->mutex);
    
    size_t result_count = 0;
    for (size_t i = 0; i < wm->capacity && result_count < max_results; i++) {
        const ce_wm_slot_t *slot = &wm->slots[i];
        
        if (slot->is_active && slot->item && slot->current_saliency >= threshold) {
            result[result_count] = slot->item;
            result_count++;
        }
    }
    
    pthread_mutex_unlock((pthread_mutex_t *)&wm->mutex);
    
    return result_count;
}

ce_error_t ce_wm_update(ce_working_memory_t *wm) {
    if (!wm) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&wm->mutex);
    
    double current_time = ce_get_timestamp();
    
    /* Update saliency for all active slots */
    for (size_t i = 0; i < wm->capacity; i++) {
        ce_wm_slot_t *slot = &wm->slots[i];
        
        if (slot->is_active && slot->item) {
            update_slot_saliency(slot, wm->decay_rate, current_time);
        }
    }
    
    /* Remove items below threshold */
    remove_low_saliency_items(wm);
    
    /* Update statistics */
    wm->total_saliency = 0.0;
    for (size_t i = 0; i < wm->capacity; i++) {
        const ce_wm_slot_t *slot = &wm->slots[i];
        if (slot->is_active) {
            wm->total_saliency += slot->current_saliency;
        }
    }
    wm->avg_saliency = wm->count > 0 ? wm->total_saliency / wm->count : 0.0;
    
    pthread_mutex_unlock(&wm->mutex);
    
    return CE_SUCCESS;
}

ce_error_t ce_wm_decay(ce_working_memory_t *wm) {
    if (!wm) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&wm->mutex);
    
    double current_time = ce_get_timestamp();
    
    /* Apply decay to all active slots */
    for (size_t i = 0; i < wm->capacity; i++) {
        ce_wm_slot_t *slot = &wm->slots[i];
        
        if (slot->is_active && slot->item) {
            update_slot_saliency(slot, wm->decay_rate, current_time);
        }
    }
    
    pthread_mutex_unlock(&wm->mutex);
    
    return CE_SUCCESS;
}

ce_error_t ce_wm_consolidate(ce_working_memory_t *wm, float similarity_threshold) {
    if (!wm) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&wm->mutex);
    
    /* Find similar items and merge them */
    for (size_t i = 0; i < wm->capacity; i++) {
        ce_wm_slot_t *slot1 = &wm->slots[i];
        
        if (!slot1->is_active || !slot1->item) {
            continue;
        }
        
        for (size_t j = i + 1; j < wm->capacity; j++) {
            ce_wm_slot_t *slot2 = &wm->slots[j];
            
            if (!slot2->is_active || !slot2->item) {
                continue;
            }
            
            float similarity = compute_item_similarity(slot1->item, slot2->item);
            
            if (similarity >= similarity_threshold) {
                /* Merge items - keep the one with higher saliency */
                if (slot1->current_saliency >= slot2->current_saliency) {
                    /* Merge slot2 into slot1 */
                    slot1->base_saliency = (slot1->base_saliency + slot2->base_saliency) / 2.0f;
                    slot1->current_saliency = slot1->base_saliency;
                    slot1->access_count += slot2->access_count;
                    
                    /* Remove slot2 */
                    if (wm->on_item_removed) {
                        wm->on_item_removed(slot2->item, wm->callback_user_data);
                    }
                    ce_item_free(slot2->item);
                    memset(slot2, 0, sizeof(ce_wm_slot_t));
                    slot2->is_active = false;
                    wm->count--;
                    wm->total_items_removed++;
                } else {
                    /* Merge slot1 into slot2 */
                    slot2->base_saliency = (slot1->base_saliency + slot2->base_saliency) / 2.0f;
                    slot2->current_saliency = slot2->base_saliency;
                    slot2->access_count += slot1->access_count;
                    
                    /* Remove slot1 */
                    if (wm->on_item_removed) {
                        wm->on_item_removed(slot1->item, wm->callback_user_data);
                    }
                    ce_item_free(slot1->item);
                    memset(slot1, 0, sizeof(ce_wm_slot_t));
                    slot1->is_active = false;
                    wm->count--;
                    wm->total_items_removed++;
                    break; /* Move to next slot1 */
                }
            }
        }
    }
    
    pthread_mutex_unlock(&wm->mutex);
    
    return CE_SUCCESS;
}

ce_error_t ce_wm_clear(ce_working_memory_t *wm) {
    if (!wm) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&wm->mutex);
    
    /* Free all items and clear slots */
    for (size_t i = 0; i < wm->capacity; i++) {
        ce_wm_slot_t *slot = &wm->slots[i];
        
        if (slot->is_active && slot->item) {
            if (wm->on_item_removed) {
                wm->on_item_removed(slot->item, wm->callback_user_data);
            }
            ce_item_free(slot->item);
            wm->total_items_removed++;
        }
        
        memset(slot, 0, sizeof(ce_wm_slot_t));
        slot->is_active = false;
    }
    
    wm->count = 0;
    wm->total_saliency = 0.0;
    wm->avg_saliency = 0.0;
    
    pthread_mutex_unlock(&wm->mutex);
    
    return CE_SUCCESS;
}

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
} *stats) {
    if (!wm || !stats) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock((pthread_mutex_t *)&wm->mutex);
    
    stats->capacity = wm->capacity;
    stats->count = wm->count;
    stats->active_slots = wm->count;
    stats->total_saliency = wm->total_saliency;
    stats->avg_saliency = wm->avg_saliency;
    stats->total_items_added = wm->total_items_added;
    stats->total_items_removed = wm->total_items_removed;
    stats->total_accesses = wm->total_accesses;
    
    /* Find min/max saliency */
    stats->max_saliency = 0.0;
    stats->min_saliency = 1.0;
    
    for (size_t i = 0; i < wm->capacity; i++) {
        const ce_wm_slot_t *slot = &wm->slots[i];
        
        if (slot->is_active) {
            if (slot->current_saliency > stats->max_saliency) {
                stats->max_saliency = slot->current_saliency;
            }
            if (slot->current_saliency < stats->min_saliency) {
                stats->min_saliency = slot->current_saliency;
            }
        }
    }
    
    pthread_mutex_unlock((pthread_mutex_t *)&wm->mutex);
    
    return CE_SUCCESS;
}

ce_error_t ce_wm_set_callbacks(ce_working_memory_t *wm,
                               void (*on_item_added)(ce_item_t *item, void *user_data),
                               void (*on_item_removed)(ce_item_t *item, void *user_data),
                               void (*on_item_accessed)(ce_item_t *item, void *user_data),
                               void *user_data) {
    if (!wm) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&wm->mutex);
    
    wm->on_item_added = on_item_added;
    wm->on_item_removed = on_item_removed;
    wm->on_item_accessed = on_item_accessed;
    wm->callback_user_data = user_data;
    
    pthread_mutex_unlock(&wm->mutex);
    
    return CE_SUCCESS;
}
