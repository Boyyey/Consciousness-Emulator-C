/**
 * Consciousness Emulator - Long-Term Memory Module
 * 
 * Implements episodic and semantic long-term memory with vector-based
 * similarity search and consolidation mechanisms.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#ifndef CE_LONG_TERM_MEMORY_H
#define CE_LONG_TERM_MEMORY_H

#include "../../include/consciousness.h"
#include <stdbool.h>

/* ============================================================================
 * LTM Configuration
 * ============================================================================ */

#define CE_LTM_DEFAULT_MAX_EPISODES 10000
#define CE_LTM_DEFAULT_EMBEDDING_DIM 64
#define CE_LTM_DEFAULT_CONSOLIDATION_THRESHOLD 0.8f
#define CE_LTM_DEFAULT_SIMILARITY_THRESHOLD 0.7f
#define CE_LTM_MAX_SEARCH_RESULTS 100

/* ============================================================================
 * Episode Structure
 * ============================================================================ */

typedef struct {
    uint64_t id;
    ce_item_t *item;                /* The original cognitive item */
    char *context;                  /* Additional context information */
    float *embedding;               /* Vector embedding */
    size_t embedding_dim;           /* Embedding dimension */
    double timestamp;               /* When the episode was stored */
    double last_accessed;           /* Last access time */
    float access_count;             /* Number of times accessed */
    float consolidation_strength;   /* How well consolidated this memory is */
    bool is_consolidated;           /* Whether this episode is consolidated */
} ce_episode_t;

/* ============================================================================
 * Semantic Index Structure
 * ============================================================================ */

typedef struct {
    uint64_t *episode_ids;          /* Mapping from index to episode ID */
    float *vectors;                 /* Vector embeddings (N x D) */
    size_t count;                   /* Number of vectors */
    size_t capacity;                /* Allocated capacity */
    size_t embedding_dim;           /* Embedding dimension */
    bool *is_active;                /* Whether each vector is active */
} ce_semantic_index_t;

/* ============================================================================
 * Consolidation System
 * ============================================================================ */

typedef struct {
    float *cluster_centers;         /* Cluster center vectors */
    uint64_t *cluster_episodes;     /* Episodes in each cluster */
    size_t *cluster_sizes;          /* Size of each cluster */
    size_t cluster_count;           /* Number of clusters */
    size_t max_clusters;            /* Maximum number of clusters */
    float consolidation_threshold;  /* Threshold for consolidation */
} ce_consolidation_system_t;

/* ============================================================================
 * Long-Term Memory Structure
 * ============================================================================ */

typedef struct ce_long_term_memory {
    /* Episodic storage */
    ce_episode_t *episodes;         /* Array of episodes */
    size_t episode_count;           /* Current number of episodes */
    size_t max_episodes;            /* Maximum number of episodes */
    
    /* Semantic index */
    ce_semantic_index_t semantic_index;
    
    /* Consolidation system */
    ce_consolidation_system_t consolidation;
    
    /* Configuration */
    size_t embedding_dim;           /* Embedding dimension */
    float similarity_threshold;     /* Similarity threshold for search */
    bool auto_consolidation;        /* Whether to auto-consolidate */
    
    /* Statistics */
    uint64_t total_episodes_stored;
    uint64_t total_searches;
    uint64_t total_consolidations;
    double total_search_time;
    double avg_search_time;
    double max_search_time;
    
    /* Callbacks */
    void (*on_episode_stored)(const ce_episode_t *episode, void *user_data);
    void (*on_episode_accessed)(const ce_episode_t *episode, void *user_data);
    void (*on_consolidation)(const ce_episode_t *episodes, size_t count, void *user_data);
    void *callback_user_data;
    
    /* Synchronization */
    pthread_mutex_t mutex;
} ce_long_term_memory_t;

/* ============================================================================
 * LTM API
 * ============================================================================ */

/**
 * Create a new long-term memory instance
 * @param embedding_dim Embedding dimension
 * @param max_episodes Maximum number of episodes
 * @return LTM instance or NULL on error
 */
ce_long_term_memory_t *ce_ltm_create(size_t embedding_dim, size_t max_episodes);

/**
 * Create LTM with custom configuration
 * @param embedding_dim Embedding dimension
 * @param max_episodes Maximum number of episodes
 * @param similarity_threshold Similarity threshold for search
 * @param auto_consolidation Whether to auto-consolidate
 * @return LTM instance or NULL on error
 */
ce_long_term_memory_t *ce_ltm_create_with_config(size_t embedding_dim, size_t max_episodes,
                                                 float similarity_threshold, bool auto_consolidation);

/**
 * Free LTM instance
 * @param ltm LTM instance
 */
void ce_ltm_free(ce_long_term_memory_t *ltm);

/**
 * Store an episode in LTM
 * @param ltm LTM instance
 * @param item Item to store
 * @param context Additional context
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_store_episode(ce_long_term_memory_t *ltm, const ce_item_t *item, const char *context);

/**
 * Search LTM by similarity
 * @param ltm LTM instance
 * @param query_embedding Query vector
 * @param k Number of results
 * @param results Output array (must be pre-allocated)
 * @return Number of results found
 */
size_t ce_ltm_search(ce_long_term_memory_t *ltm, const float *query_embedding, 
                     size_t k, ce_item_t **results);

/**
 * Search LTM by content similarity
 * @param ltm LTM instance
 * @param query_item Query item
 * @param k Number of results
 * @param results Output array (must be pre-allocated)
 * @return Number of results found
 */
size_t ce_ltm_search_by_item(ce_long_term_memory_t *ltm, const ce_item_t *query_item,
                             size_t k, ce_item_t **results);

/**
 * Access an episode (update access statistics)
 * @param ltm LTM instance
 * @param episode_id Episode ID
 * @return Episode or NULL if not found
 */
ce_episode_t *ce_ltm_access_episode(ce_long_term_memory_t *ltm, uint64_t episode_id);

/**
 * Consolidate memories
 * @param ltm LTM instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_consolidate(ce_long_term_memory_t *ltm);

/**
 * Get LTM statistics
 * @param ltm LTM instance
 * @param stats Output statistics structure
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_get_stats(const ce_long_term_memory_t *ltm, struct {
    size_t total_episodes;
    size_t consolidated_episodes;
    size_t semantic_index_size;
    size_t cluster_count;
    uint64_t total_searches;
    uint64_t total_consolidations;
    double avg_search_time;
    double max_search_time;
} *stats);

/**
 * Set callback functions
 * @param ltm LTM instance
 * @param on_episode_stored Callback for episode storage
 * @param on_episode_accessed Callback for episode access
 * @param on_consolidation Callback for consolidation
 * @param user_data User data passed to callbacks
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_set_callbacks(ce_long_term_memory_t *ltm,
                                void (*on_episode_stored)(const ce_episode_t *episode, void *user_data),
                                void (*on_episode_accessed)(const ce_episode_t *episode, void *user_data),
                                void (*on_consolidation)(const ce_episode_t *episodes, size_t count, void *user_data),
                                void *user_data);

/* ============================================================================
 * Episode Management
 * ============================================================================ */

/**
 * Create a new episode
 * @param item Original item
 * @param context Additional context
 * @param embedding Vector embedding
 * @param embedding_dim Embedding dimension
 * @return New episode or NULL on error
 */
ce_episode_t *ce_episode_create(const ce_item_t *item, const char *context,
                               const float *embedding, size_t embedding_dim);

/**
 * Clone an episode
 * @param episode Source episode
 * @return Cloned episode or NULL on error
 */
ce_episode_t *ce_episode_clone(const ce_episode_t *episode);

/**
 * Free an episode
 * @param episode Episode to free
 */
void ce_episode_free(ce_episode_t *episode);

/**
 * Update episode access statistics
 * @param episode Episode to update
 */
void ce_episode_update_access(ce_episode_t *episode);

/* ============================================================================
 * Semantic Index Operations
 * ============================================================================ */

/**
 * Add episode to semantic index
 * @param ltm LTM instance
 * @param episode Episode to add
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_add_to_semantic_index(ce_long_term_memory_t *ltm, const ce_episode_t *episode);

/**
 * Remove episode from semantic index
 * @param ltm LTM instance
 * @param episode_id Episode ID to remove
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_remove_from_semantic_index(ce_long_term_memory_t *ltm, uint64_t episode_id);

/**
 * Search semantic index
 * @param ltm LTM instance
 * @param query_embedding Query vector
 * @param k Number of results
 * @param results Output array of episode IDs
 * @param similarities Output array of similarities
 * @return Number of results found
 */
size_t ce_ltm_search_semantic_index(ce_long_term_memory_t *ltm, const float *query_embedding,
                                   size_t k, uint64_t *results, float *similarities);

/* ============================================================================
 * Consolidation Operations
 * ============================================================================ */

/**
 * Initialize consolidation system
 * @param ltm LTM instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_init_consolidation(ce_long_term_memory_t *ltm);

/**
 * Perform k-means clustering for consolidation
 * @param ltm LTM instance
 * @param k Number of clusters
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_perform_clustering(ce_long_term_memory_t *ltm, size_t k);

/**
 * Merge similar episodes
 * @param ltm LTM instance
 * @param episode_ids Array of episode IDs to merge
 * @param count Number of episodes to merge
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_merge_episodes(ce_long_term_memory_t *ltm, const uint64_t *episode_ids, size_t count);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Compute episode similarity
 * @param episode1 First episode
 * @param episode2 Second episode
 * @return Similarity score [0.0, 1.0]
 */
float ce_ltm_compute_episode_similarity(const ce_episode_t *episode1, const ce_episode_t *episode2);

/**
 * Generate embedding for an item (placeholder implementation)
 * @param item Item to embed
 * @param embedding Output embedding
 * @param embedding_dim Embedding dimension
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_generate_embedding(const ce_item_t *item, float *embedding, size_t embedding_dim);

/**
 * Serialize LTM to file
 * @param ltm LTM instance
 * @param filename Output filename
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_serialize(const ce_long_term_memory_t *ltm, const char *filename);

/**
 * Deserialize LTM from file
 * @param ltm LTM instance
 * @param filename Input filename
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_ltm_deserialize(ce_long_term_memory_t *ltm, const char *filename);

#endif /* CE_LONG_TERM_MEMORY_H */
