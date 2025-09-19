/**
 * Consciousness Emulator - Long-Term Memory Implementation
 * 
 * Implements episodic and semantic long-term memory with vector-based
 * similarity search and consolidation mechanisms.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "long_term_memory.h"
#include "../utils/math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * Initialize semantic index
 */
static ce_error_t init_semantic_index(ce_semantic_index_t *index, size_t capacity, size_t embedding_dim) {
    index->episode_ids = calloc(capacity, sizeof(uint64_t));
    if (!index->episode_ids) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    index->vectors = calloc(capacity * embedding_dim, sizeof(float));
    if (!index->vectors) {
        free(index->episode_ids);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    index->is_active = calloc(capacity, sizeof(bool));
    if (!index->is_active) {
        free(index->vectors);
        free(index->episode_ids);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    index->count = 0;
    index->capacity = capacity;
    index->embedding_dim = embedding_dim;
    
    return CE_SUCCESS;
}

/**
 * Cleanup semantic index
 */
static void cleanup_semantic_index(ce_semantic_index_t *index) {
    if (index->episode_ids) {
        free(index->episode_ids);
    }
    if (index->vectors) {
        free(index->vectors);
    }
    if (index->is_active) {
        free(index->is_active);
    }
    memset(index, 0, sizeof(ce_semantic_index_t));
}

/**
 * Initialize consolidation system
 */
static ce_error_t init_consolidation_system(ce_consolidation_system_t *consolidation, 
                                           size_t max_clusters, size_t embedding_dim) {
    consolidation->cluster_centers = calloc(max_clusters * embedding_dim, sizeof(float));
    if (!consolidation->cluster_centers) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    consolidation->cluster_episodes = calloc(max_clusters, sizeof(uint64_t));
    if (!consolidation->cluster_episodes) {
        free(consolidation->cluster_centers);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    consolidation->cluster_sizes = calloc(max_clusters, sizeof(size_t));
    if (!consolidation->cluster_sizes) {
        free(consolidation->cluster_episodes);
        free(consolidation->cluster_centers);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    consolidation->cluster_count = 0;
    consolidation->max_clusters = max_clusters;
    consolidation->consolidation_threshold = CE_LTM_DEFAULT_CONSOLIDATION_THRESHOLD;
    
    return CE_SUCCESS;
}

/**
 * Cleanup consolidation system
 */
static void cleanup_consolidation_system(ce_consolidation_system_t *consolidation) {
    if (consolidation->cluster_centers) {
        free(consolidation->cluster_centers);
    }
    if (consolidation->cluster_episodes) {
        free(consolidation->cluster_episodes);
    }
    if (consolidation->cluster_sizes) {
        free(consolidation->cluster_sizes);
    }
    memset(consolidation, 0, sizeof(ce_consolidation_system_t));
}

/**
 * Find episode by ID
 */
static ce_episode_t *find_episode_by_id(ce_long_term_memory_t *ltm, uint64_t episode_id) {
    for (size_t i = 0; i < ltm->episode_count; i++) {
        if (ltm->episodes[i].id == episode_id) {
            return &ltm->episodes[i];
        }
    }
    return NULL;
}

/**
 * Find free slot in episodes array
 */
static size_t find_free_episode_slot(ce_long_term_memory_t *ltm) {
    for (size_t i = 0; i < ltm->max_episodes; i++) {
        if (ltm->episodes[i].id == 0) { /* Empty slot */
            return i;
        }
    }
    return SIZE_MAX; /* No free slots */
}

/**
 * Resize episodes array if needed
 */
static ce_error_t resize_episodes_array(ce_long_term_memory_t *ltm) {
    if (ltm->episode_count < ltm->max_episodes) {
        return CE_SUCCESS; /* No need to resize */
    }
    
    size_t new_capacity = ltm->max_episodes * 2;
    ce_episode_t *new_episodes = realloc(ltm->episodes, new_capacity * sizeof(ce_episode_t));
    if (!new_episodes) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Initialize new slots */
    for (size_t i = ltm->max_episodes; i < new_capacity; i++) {
        memset(&new_episodes[i], 0, sizeof(ce_episode_t));
    }
    
    ltm->episodes = new_episodes;
    ltm->max_episodes = new_capacity;
    
    return CE_SUCCESS;
}

/* ============================================================================
 * Public API Implementation
 * ============================================================================ */

ce_long_term_memory_t *ce_ltm_create(size_t embedding_dim, size_t max_episodes) {
    return ce_ltm_create_with_config(embedding_dim, max_episodes, 
                                    CE_LTM_DEFAULT_SIMILARITY_THRESHOLD, true);
}

ce_long_term_memory_t *ce_ltm_create_with_config(size_t embedding_dim, size_t max_episodes,
                                                 float similarity_threshold, bool auto_consolidation) {
    if (embedding_dim == 0 || max_episodes == 0) {
        return NULL;
    }
    
    ce_long_term_memory_t *ltm = calloc(1, sizeof(ce_long_term_memory_t));
    if (!ltm) {
        return NULL;
    }
    
    /* Initialize basic configuration */
    ltm->embedding_dim = embedding_dim;
    ltm->max_episodes = max_episodes;
    ltm->similarity_threshold = similarity_threshold;
    ltm->auto_consolidation = auto_consolidation;
    
    /* Initialize episodes array */
    ltm->episodes = calloc(max_episodes, sizeof(ce_episode_t));
    if (!ltm->episodes) {
        free(ltm);
        return NULL;
    }
    
    ltm->episode_count = 0;
    
    /* Initialize semantic index */
    if (init_semantic_index(&ltm->semantic_index, max_episodes, embedding_dim) != CE_SUCCESS) {
        free(ltm->episodes);
        free(ltm);
        return NULL;
    }
    
    /* Initialize consolidation system */
    if (init_consolidation_system(&ltm->consolidation, max_episodes / 10, embedding_dim) != CE_SUCCESS) {
        cleanup_semantic_index(&ltm->semantic_index);
        free(ltm->episodes);
        free(ltm);
        return NULL;
    }
    
    /* Initialize statistics */
    ltm->total_episodes_stored = 0;
    ltm->total_searches = 0;
    ltm->total_consolidations = 0;
    ltm->total_search_time = 0.0;
    ltm->avg_search_time = 0.0;
    ltm->max_search_time = 0.0;
    
    /* Initialize callbacks */
    ltm->on_episode_stored = NULL;
    ltm->on_episode_accessed = NULL;
    ltm->on_consolidation = NULL;
    ltm->callback_user_data = NULL;
    
    /* Initialize synchronization */
    if (pthread_mutex_init(&ltm->mutex, NULL) != 0) {
        cleanup_consolidation_system(&ltm->consolidation);
        cleanup_semantic_index(&ltm->semantic_index);
        free(ltm->episodes);
        free(ltm);
        return NULL;
    }
    
    return ltm;
}

void ce_ltm_free(ce_long_term_memory_t *ltm) {
    if (!ltm) {
        return;
    }
    
    /* Free all episodes */
    for (size_t i = 0; i < ltm->max_episodes; i++) {
        ce_episode_free(&ltm->episodes[i]);
    }
    
    /* Cleanup subsystems */
    cleanup_consolidation_system(&ltm->consolidation);
    cleanup_semantic_index(&ltm->semantic_index);
    
    /* Destroy synchronization */
    pthread_mutex_destroy(&ltm->mutex);
    
    /* Free memory */
    free(ltm->episodes);
    free(ltm);
}

ce_error_t ce_ltm_store_episode(ce_long_term_memory_t *ltm, const ce_item_t *item, const char *context) {
    if (!ltm || !item) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&ltm->mutex);
    
    /* Resize episodes array if needed */
    if (resize_episodes_array(ltm) != CE_SUCCESS) {
        pthread_mutex_unlock(&ltm->mutex);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Find free slot */
    size_t slot = find_free_episode_slot(ltm);
    if (slot == SIZE_MAX) {
        pthread_mutex_unlock(&ltm->mutex);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Create episode */
    ce_episode_t *episode = &ltm->episodes[slot];
    
    static uint64_t episode_id_counter = 1;
    episode->id = episode_id_counter++;
    episode->item = ce_item_clone(item);
    if (!episode->item) {
        pthread_mutex_unlock(&ltm->mutex);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Copy context */
    if (context) {
        episode->context = malloc(strlen(context) + 1);
        if (!episode->context) {
            ce_item_free(episode->item);
            pthread_mutex_unlock(&ltm->mutex);
            return CE_ERROR_OUT_OF_MEMORY;
        }
        strcpy(episode->context, context);
    } else {
        episode->context = NULL;
    }
    
    /* Generate or copy embedding */
    episode->embedding = malloc(ltm->embedding_dim * sizeof(float));
    if (!episode->embedding) {
        if (episode->context) free(episode->context);
        ce_item_free(episode->item);
        pthread_mutex_unlock(&ltm->mutex);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    if (item->embedding && item->embedding_dim == ltm->embedding_dim) {
        memcpy(episode->embedding, item->embedding, ltm->embedding_dim * sizeof(float));
    } else {
        ce_ltm_generate_embedding(item, episode->embedding, ltm->embedding_dim);
    }
    
    episode->embedding_dim = ltm->embedding_dim;
    episode->timestamp = ce_get_timestamp();
    episode->last_accessed = episode->timestamp;
    episode->access_count = 0.0f;
    episode->consolidation_strength = 0.0f;
    episode->is_consolidated = false;
    
    ltm->episode_count++;
    ltm->total_episodes_stored++;
    
    /* Add to semantic index */
    ce_ltm_add_to_semantic_index(ltm, episode);
    
    /* Call storage callback */
    if (ltm->on_episode_stored) {
        ltm->on_episode_stored(episode, ltm->callback_user_data);
    }
    
    /* Auto-consolidate if enabled */
    if (ltm->auto_consolidation && ltm->episode_count % 100 == 0) {
        ce_ltm_consolidate(ltm);
    }
    
    pthread_mutex_unlock(&ltm->mutex);
    
    return CE_SUCCESS;
}

size_t ce_ltm_search(ce_long_term_memory_t *ltm, const float *query_embedding, 
                     size_t k, ce_item_t **results) {
    if (!ltm || !query_embedding || !results || k == 0) {
        return 0;
    }
    
    pthread_mutex_lock(&ltm->mutex);
    
    double search_start = ce_get_timestamp();
    
    /* Search semantic index */
    uint64_t *episode_ids = malloc(k * sizeof(uint64_t));
    float *similarities = malloc(k * sizeof(float));
    
    if (!episode_ids || !similarities) {
        free(episode_ids);
        free(similarities);
        pthread_mutex_unlock(&ltm->mutex);
        return 0;
    }
    
    size_t found_count = ce_ltm_search_semantic_index(ltm, query_embedding, k, episode_ids, similarities);
    
    /* Convert episode IDs to items */
    size_t result_count = 0;
    for (size_t i = 0; i < found_count && result_count < k; i++) {
        ce_episode_t *episode = find_episode_by_id(ltm, episode_ids[i]);
        if (episode && episode->item) {
            results[result_count] = episode->item;
            result_count++;
            
            /* Update access statistics */
            ce_episode_update_access(episode);
        }
    }
    
    /* Update search statistics */
    double search_end = ce_get_timestamp();
    double search_time = search_end - search_start;
    
    ltm->total_searches++;
    ltm->total_search_time += search_time;
    ltm->avg_search_time = ltm->total_search_time / ltm->total_searches;
    
    if (search_time > ltm->max_search_time) {
        ltm->max_search_time = search_time;
    }
    
    free(episode_ids);
    free(similarities);
    
    pthread_mutex_unlock(&ltm->mutex);
    
    return result_count;
}

size_t ce_ltm_search_by_item(ce_long_term_memory_t *ltm, const ce_item_t *query_item,
                             size_t k, ce_item_t **results) {
    if (!ltm || !query_item || !results || k == 0) {
        return 0;
    }
    
    /* Generate embedding for query item if it doesn't have one */
    float *query_embedding = malloc(ltm->embedding_dim * sizeof(float));
    if (!query_embedding) {
        return 0;
    }
    
    if (query_item->embedding && query_item->embedding_dim == ltm->embedding_dim) {
        memcpy(query_embedding, query_item->embedding, ltm->embedding_dim * sizeof(float));
    } else {
        ce_ltm_generate_embedding(query_item, query_embedding, ltm->embedding_dim);
    }
    
    size_t result_count = ce_ltm_search(ltm, query_embedding, k, results);
    
    free(query_embedding);
    return result_count;
}

ce_episode_t *ce_ltm_access_episode(ce_long_term_memory_t *ltm, uint64_t episode_id) {
    if (!ltm) {
        return NULL;
    }
    
    pthread_mutex_lock(&ltm->mutex);
    
    ce_episode_t *episode = find_episode_by_id(ltm, episode_id);
    if (episode) {
        ce_episode_update_access(episode);
        
        /* Call access callback */
        if (ltm->on_episode_accessed) {
            ltm->on_episode_accessed(episode, ltm->callback_user_data);
        }
    }
    
    pthread_mutex_unlock(&ltm->mutex);
    
    return episode;
}

ce_error_t ce_ltm_consolidate(ce_long_term_memory_t *ltm) {
    if (!ltm) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&ltm->mutex);
    
    /* Simple consolidation: merge very similar episodes */
    size_t consolidated_count = 0;
    
    for (size_t i = 0; i < ltm->episode_count; i++) {
        ce_episode_t *episode1 = &ltm->episodes[i];
        if (!episode1->item || episode1->is_consolidated) {
            continue;
        }
        
        for (size_t j = i + 1; j < ltm->episode_count; j++) {
            ce_episode_t *episode2 = &ltm->episodes[j];
            if (!episode2->item || episode2->is_consolidated) {
                continue;
            }
            
            float similarity = ce_ltm_compute_episode_similarity(episode1, episode2);
            
            if (similarity >= ltm->consolidation.consolidation_threshold) {
                /* Merge episodes - keep the one with higher access count */
                if (episode1->access_count >= episode2->access_count) {
                    /* Merge episode2 into episode1 */
                    episode1->access_count += episode2->access_count;
                    episode1->consolidation_strength += 1.0f;
                    
                    /* Remove episode2 from semantic index */
                    ce_ltm_remove_from_semantic_index(ltm, episode2->id);
                    
                    /* Free episode2 */
                    ce_episode_free(episode2);
                    memset(episode2, 0, sizeof(ce_episode_t));
                    
                    ltm->episode_count--;
                    consolidated_count++;
                } else {
                    /* Merge episode1 into episode2 */
                    episode2->access_count += episode1->access_count;
                    episode2->consolidation_strength += 1.0f;
                    
                    /* Remove episode1 from semantic index */
                    ce_ltm_remove_from_semantic_index(ltm, episode1->id);
                    
                    /* Free episode1 */
                    ce_episode_free(episode1);
                    memset(episode1, 0, sizeof(ce_episode_t));
                    
                    ltm->episode_count--;
                    consolidated_count++;
                    break; /* Move to next episode1 */
                }
            }
        }
    }
    
    ltm->total_consolidations++;
    
    /* Call consolidation callback */
    if (ltm->on_consolidation && consolidated_count > 0) {
        ltm->on_consolidation(NULL, consolidated_count, ltm->callback_user_data);
    }
    
    pthread_mutex_unlock(&ltm->mutex);
    
    return CE_SUCCESS;
}

ce_error_t ce_ltm_get_stats(const ce_long_term_memory_t *ltm, struct {
    size_t total_episodes;
    size_t consolidated_episodes;
    size_t semantic_index_size;
    size_t cluster_count;
    uint64_t total_searches;
    uint64_t total_consolidations;
    double avg_search_time;
    double max_search_time;
} *stats) {
    if (!ltm || !stats) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock((pthread_mutex_t *)&ltm->mutex);
    
    stats->total_episodes = ltm->episode_count;
    stats->semantic_index_size = ltm->semantic_index.count;
    stats->cluster_count = ltm->consolidation.cluster_count;
    stats->total_searches = ltm->total_searches;
    stats->total_consolidations = ltm->total_consolidations;
    stats->avg_search_time = ltm->avg_search_time;
    stats->max_search_time = ltm->max_search_time;
    
    /* Count consolidated episodes */
    stats->consolidated_episodes = 0;
    for (size_t i = 0; i < ltm->max_episodes; i++) {
        if (ltm->episodes[i].is_consolidated) {
            stats->consolidated_episodes++;
        }
    }
    
    pthread_mutex_unlock((pthread_mutex_t *)&ltm->mutex);
    
    return CE_SUCCESS;
}

ce_error_t ce_ltm_set_callbacks(ce_long_term_memory_t *ltm,
                                void (*on_episode_stored)(const ce_episode_t *episode, void *user_data),
                                void (*on_episode_accessed)(const ce_episode_t *episode, void *user_data),
                                void (*on_consolidation)(const ce_episode_t *episodes, size_t count, void *user_data),
                                void *user_data) {
    if (!ltm) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&ltm->mutex);
    
    ltm->on_episode_stored = on_episode_stored;
    ltm->on_episode_accessed = on_episode_accessed;
    ltm->on_consolidation = on_consolidation;
    ltm->callback_user_data = user_data;
    
    pthread_mutex_unlock(&ltm->mutex);
    
    return CE_SUCCESS;
}

/* ============================================================================
 * Episode Management Implementation
 * ============================================================================ */

ce_episode_t *ce_episode_create(const ce_item_t *item, const char *context,
                               const float *embedding, size_t embedding_dim) {
    if (!item || !embedding || embedding_dim == 0) {
        return NULL;
    }
    
    ce_episode_t *episode = calloc(1, sizeof(ce_episode_t));
    if (!episode) {
        return NULL;
    }
    
    static uint64_t episode_id_counter = 1;
    episode->id = episode_id_counter++;
    
    episode->item = ce_item_clone(item);
    if (!episode->item) {
        free(episode);
        return NULL;
    }
    
    if (context) {
        episode->context = malloc(strlen(context) + 1);
        if (!episode->context) {
            ce_item_free(episode->item);
            free(episode);
            return NULL;
        }
        strcpy(episode->context, context);
    }
    
    episode->embedding = malloc(embedding_dim * sizeof(float));
    if (!episode->embedding) {
        if (episode->context) free(episode->context);
        ce_item_free(episode->item);
        free(episode);
        return NULL;
    }
    
    memcpy(episode->embedding, embedding, embedding_dim * sizeof(float));
    episode->embedding_dim = embedding_dim;
    episode->timestamp = ce_get_timestamp();
    episode->last_accessed = episode->timestamp;
    episode->access_count = 0.0f;
    episode->consolidation_strength = 0.0f;
    episode->is_consolidated = false;
    
    return episode;
}

ce_episode_t *ce_episode_clone(const ce_episode_t *episode) {
    if (!episode) {
        return NULL;
    }
    
    ce_episode_t *clone = malloc(sizeof(ce_episode_t));
    if (!clone) {
        return NULL;
    }
    
    *clone = *episode;
    
    /* Clone item */
    if (episode->item) {
        clone->item = ce_item_clone(episode->item);
        if (!clone->item) {
            free(clone);
            return NULL;
        }
    }
    
    /* Clone context */
    if (episode->context) {
        clone->context = malloc(strlen(episode->context) + 1);
        if (!clone->context) {
            ce_item_free(clone->item);
            free(clone);
            return NULL;
        }
        strcpy(clone->context, episode->context);
    }
    
    /* Clone embedding */
    if (episode->embedding && episode->embedding_dim > 0) {
        clone->embedding = malloc(episode->embedding_dim * sizeof(float));
        if (!clone->embedding) {
            if (clone->context) free(clone->context);
            ce_item_free(clone->item);
            free(clone);
            return NULL;
        }
        memcpy(clone->embedding, episode->embedding, episode->embedding_dim * sizeof(float));
    }
    
    return clone;
}

void ce_episode_free(ce_episode_t *episode) {
    if (!episode) {
        return;
    }
    
    if (episode->item) {
        ce_item_free(episode->item);
    }
    
    if (episode->context) {
        free(episode->context);
    }
    
    if (episode->embedding) {
        free(episode->embedding);
    }
    
    memset(episode, 0, sizeof(ce_episode_t));
}

void ce_episode_update_access(ce_episode_t *episode) {
    if (!episode) {
        return;
    }
    
    episode->last_accessed = ce_get_timestamp();
    episode->access_count += 1.0f;
}

/* ============================================================================
 * Semantic Index Implementation
 * ============================================================================ */

ce_error_t ce_ltm_add_to_semantic_index(ce_long_term_memory_t *ltm, const ce_episode_t *episode) {
    if (!ltm || !episode || !episode->embedding) {
        return CE_ERROR_NULL_POINTER;
    }
    
    ce_semantic_index_t *index = &ltm->semantic_index;
    
    if (index->count >= index->capacity) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Add episode to index */
    index->episode_ids[index->count] = episode->id;
    index->is_active[index->count] = true;
    
    /* Copy embedding */
    memcpy(&index->vectors[index->count * index->embedding_dim], 
           episode->embedding, index->embedding_dim * sizeof(float));
    
    index->count++;
    
    return CE_SUCCESS;
}

ce_error_t ce_ltm_remove_from_semantic_index(ce_long_term_memory_t *ltm, uint64_t episode_id) {
    if (!ltm) {
        return CE_ERROR_NULL_POINTER;
    }
    
    ce_semantic_index_t *index = &ltm->semantic_index;
    
    for (size_t i = 0; i < index->count; i++) {
        if (index->episode_ids[i] == episode_id) {
            /* Mark as inactive */
            index->is_active[i] = false;
            return CE_SUCCESS;
        }
    }
    
    return CE_ERROR_UNKNOWN; /* Episode not found */
}

size_t ce_ltm_search_semantic_index(ce_long_term_memory_t *ltm, const float *query_embedding,
                                   size_t k, uint64_t *results, float *similarities) {
    if (!ltm || !query_embedding || !results || !similarities || k == 0) {
        return 0;
    }
    
    ce_semantic_index_t *index = &ltm->semantic_index;
    
    /* Create array of similarities */
    struct {
        size_t index;
        float similarity;
    } *similarity_array = malloc(index->count * sizeof(*similarity_array));
    
    if (!similarity_array) {
        return 0;
    }
    
    /* Compute similarities */
    size_t valid_count = 0;
    for (size_t i = 0; i < index->count; i++) {
        if (index->is_active[i]) {
            float *vector = &index->vectors[i * index->embedding_dim];
            float similarity = ce_cosine_similarity(query_embedding, vector, index->embedding_dim);
            
            similarity_array[valid_count].index = i;
            similarity_array[valid_count].similarity = similarity;
            valid_count++;
        }
    }
    
    /* Sort by similarity (simple bubble sort) */
    for (size_t i = 0; i < valid_count - 1; i++) {
        for (size_t j = 0; j < valid_count - i - 1; j++) {
            if (similarity_array[j].similarity < similarity_array[j + 1].similarity) {
                struct { size_t index; float similarity; } temp = similarity_array[j];
                similarity_array[j] = similarity_array[j + 1];
                similarity_array[j + 1] = temp;
            }
        }
    }
    
    /* Return top k results */
    size_t result_count = (k < valid_count) ? k : valid_count;
    for (size_t i = 0; i < result_count; i++) {
        size_t idx = similarity_array[i].index;
        results[i] = index->episode_ids[idx];
        similarities[i] = similarity_array[i].similarity;
    }
    
    free(similarity_array);
    return result_count;
}

/* ============================================================================
 * Utility Functions Implementation
 * ============================================================================ */

float ce_ltm_compute_episode_similarity(const ce_episode_t *episode1, const ce_episode_t *episode2) {
    if (!episode1 || !episode2) {
        return 0.0f;
    }
    
    /* Use embedding similarity if available */
    if (episode1->embedding && episode2->embedding && 
        episode1->embedding_dim == episode2->embedding_dim) {
        return ce_cosine_similarity(episode1->embedding, episode2->embedding, 
                                   episode1->embedding_dim);
    }
    
    /* Fallback to content similarity */
    if (episode1->item && episode2->item && 
        episode1->item->content && episode2->item->content) {
        
        if (strcmp(episode1->item->content, episode2->item->content) == 0) {
            return 1.0f;
        }
        
        /* Simple word overlap similarity */
        char *content1 = strdup(episode1->item->content);
        char *content2 = strdup(episode2->item->content);
        
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

ce_error_t ce_ltm_generate_embedding(const ce_item_t *item, float *embedding, size_t embedding_dim) {
    if (!item || !embedding || embedding_dim == 0) {
        return CE_ERROR_NULL_POINTER;
    }
    
    /* Simple hash-based embedding generation */
    uint32_t hash = 0;
    
    /* Hash based on content */
    if (item->content) {
        for (const char *p = item->content; *p; p++) {
            hash = hash * 31 + *p;
        }
    }
    
    /* Hash based on type */
    hash = hash * 31 + item->type;
    
    /* Hash based on confidence */
    hash = hash * 31 + (uint32_t)(item->confidence * 1000);
    
    /* Generate embedding using hash as seed */
    ce_generate_random_embedding(embedding, embedding_dim, hash);
    
    return CE_SUCCESS;
}

ce_error_t ce_ltm_serialize(const ce_long_term_memory_t *ltm, const char *filename) {
    /* Placeholder implementation - would serialize to binary format */
    (void)ltm;
    (void)filename;
    return CE_ERROR_UNKNOWN;
}

ce_error_t ce_ltm_deserialize(ce_long_term_memory_t *ltm, const char *filename) {
    /* Placeholder implementation - would deserialize from binary format */
    (void)ltm;
    (void)filename;
    return CE_ERROR_UNKNOWN;
}
