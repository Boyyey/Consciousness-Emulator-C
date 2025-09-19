/**
 * Consciousness Emulator - Global Workspace Theory Module
 * 
 * Implements the Global Workspace Theory (GWT) as the central attention
 * and arbitration mechanism for the consciousness emulator.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#ifndef CE_WORKSPACE_H
#define CE_WORKSPACE_H

#include "../../include/consciousness.h"
#include "../wm/working_memory.h"
#include <stdbool.h>

/* ============================================================================
 * Workspace Configuration
 * ============================================================================ */

#define CE_WORKSPACE_DEFAULT_THRESHOLD 0.3f
#define CE_WORKSPACE_MAX_BROADCAST_ITEMS 8
#define CE_WORKSPACE_DEFAULT_ATTENTION_WEIGHT 0.4f
#define CE_WORKSPACE_DEFAULT_NOVELTY_WEIGHT 0.3f
#define CE_WORKSPACE_DEFAULT_GOAL_WEIGHT 0.2f
#define CE_WORKSPACE_DEFAULT_UNCERTAINTY_WEIGHT 0.1f

/* ============================================================================
 * Attention Mechanisms
 * ============================================================================ */

typedef enum {
    CE_ATTENTION_MODE_WINNER_TAKE_ALL = 0,
    CE_ATTENTION_MODE_TOP_K,
    CE_ATTENTION_MODE_THRESHOLD,
    CE_ATTENTION_MODE_WEIGHTED
} ce_attention_mode_t;

typedef struct {
    float attention_weight;         /* Weight for attention-based saliency */
    float novelty_weight;           /* Weight for novelty-based saliency */
    float goal_weight;              /* Weight for goal-relevance saliency */
    float uncertainty_weight;       /* Weight for uncertainty-based saliency */
    float recency_weight;           /* Weight for recency-based saliency */
} ce_saliency_weights_t;

/* ============================================================================
 * Goal System
 * ============================================================================ */

typedef struct {
    uint64_t id;
    char *description;
    float priority;
    float *embedding;
    size_t embedding_dim;
    double created_at;
    double deadline;
    bool is_active;
} ce_goal_t;

typedef struct {
    ce_goal_t *goals;
    size_t count;
    size_t capacity;
    float *current_goal_vector;     /* Current goal state as embedding */
    size_t goal_vector_dim;
} ce_goal_system_t;

/* ============================================================================
 * Workspace Structure
 * ============================================================================ */

typedef struct ce_workspace {
    ce_working_memory_t *wm;        /* Reference to working memory */
    
    /* Configuration */
    float threshold;                 /* Broadcast threshold */
    ce_attention_mode_t attention_mode;
    size_t max_broadcast_items;
    ce_saliency_weights_t saliency_weights;
    
    /* Current state */
    ce_item_list_t *broadcast_items; /* Currently broadcasted items */
    uint64_t current_broadcast_id;
    double last_broadcast_time;
    
    /* Goal system */
    ce_goal_system_t goal_system;
    
    /* Statistics */
    uint64_t total_broadcasts;
    uint64_t total_arbitrations;
    double total_arbitration_time;
    double avg_arbitration_time;
    double max_arbitration_time;
    
    /* Callbacks */
    void (*on_broadcast)(const ce_item_t *item, void *user_data);
    void (*on_arbitration)(const ce_item_list_t *selected_items, void *user_data);
    void *callback_user_data;
    
    /* Synchronization */
    pthread_mutex_t mutex;
} ce_workspace_t;

/* ============================================================================
 * Workspace API
 * ============================================================================ */

/**
 * Create a new workspace instance
 * @param wm Working memory reference
 * @param threshold Broadcast threshold
 * @return Workspace instance or NULL on error
 */
ce_workspace_t *ce_workspace_create(ce_working_memory_t *wm, float threshold);

/**
 * Create workspace with custom configuration
 * @param wm Working memory reference
 * @param threshold Broadcast threshold
 * @param attention_mode Attention mode
 * @param max_broadcast_items Maximum items to broadcast
 * @param saliency_weights Saliency computation weights
 * @return Workspace instance or NULL on error
 */
ce_workspace_t *ce_workspace_create_with_config(ce_working_memory_t *wm, float threshold,
                                               ce_attention_mode_t attention_mode,
                                               size_t max_broadcast_items,
                                               const ce_saliency_weights_t *saliency_weights);

/**
 * Free workspace instance
 * @param workspace Workspace instance
 */
void ce_workspace_free(ce_workspace_t *workspace);

/**
 * Process workspace (attention, arbitration, broadcast)
 * @param workspace Workspace instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_workspace_process(ce_workspace_t *workspace);

/**
 * Get currently broadcasted items
 * @param workspace Workspace instance
 * @return Item list (do not free, owned by workspace)
 */
const ce_item_list_t *ce_workspace_get_broadcast(const ce_workspace_t *workspace);

/**
 * Set attention mode
 * @param workspace Workspace instance
 * @param mode Attention mode
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_workspace_set_attention_mode(ce_workspace_t *workspace, ce_attention_mode_t mode);

/**
 * Set saliency weights
 * @param workspace Workspace instance
 * @param weights Saliency weights
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_workspace_set_saliency_weights(ce_workspace_t *workspace, 
                                            const ce_saliency_weights_t *weights);

/**
 * Get workspace statistics
 * @param workspace Workspace instance
 * @param stats Output statistics structure
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_workspace_get_stats(const ce_workspace_t *workspace, struct {
    uint64_t total_broadcasts;
    uint64_t total_arbitrations;
    double avg_arbitration_time;
    double max_arbitration_time;
    size_t current_broadcast_count;
    size_t active_goals;
} *stats);

/**
 * Set callback functions
 * @param workspace Workspace instance
 * @param on_broadcast Broadcast callback
 * @param on_arbitration Arbitration callback
 * @param user_data User data passed to callbacks
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_workspace_set_callbacks(ce_workspace_t *workspace,
                                     void (*on_broadcast)(const ce_item_t *item, void *user_data),
                                     void (*on_arbitration)(const ce_item_list_t *selected_items, void *user_data),
                                     void *user_data);

/* ============================================================================
 * Goal System API
 * ============================================================================ */

/**
 * Add a goal to the workspace
 * @param workspace Workspace instance
 * @param description Goal description
 * @param priority Goal priority
 * @param embedding Goal embedding (optional)
 * @param embedding_dim Embedding dimension
 * @param deadline Goal deadline (0 = no deadline)
 * @return Goal ID or 0 on error
 */
uint64_t ce_workspace_add_goal(ce_workspace_t *workspace, const char *description,
                              float priority, const float *embedding, size_t embedding_dim,
                              double deadline);

/**
 * Remove a goal from the workspace
 * @param workspace Workspace instance
 * @param goal_id Goal ID
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_workspace_remove_goal(ce_workspace_t *workspace, uint64_t goal_id);

/**
 * Update goal priority
 * @param workspace Workspace instance
 * @param goal_id Goal ID
 * @param priority New priority
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_workspace_update_goal_priority(ce_workspace_t *workspace, uint64_t goal_id, float priority);

/**
 * Get active goals
 * @param workspace Workspace instance
 * @param goals Output array (must be pre-allocated)
 * @param max_goals Maximum number of goals to return
 * @return Number of goals returned
 */
size_t ce_workspace_get_active_goals(const ce_workspace_t *workspace, ce_goal_t **goals, size_t max_goals);

/**
 * Update current goal vector
 * @param workspace Workspace instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_workspace_update_goal_vector(ce_workspace_t *workspace);

/* ============================================================================
 * Internal Functions
 * ============================================================================ */

/**
 * Compute enhanced saliency for an item
 * @param workspace Workspace instance
 * @param item Item to compute saliency for
 * @param current_time Current timestamp
 * @return Enhanced saliency score
 */
float ce_workspace_compute_saliency(const ce_workspace_t *workspace, const ce_item_t *item, double current_time);

/**
 * Compute attention-based saliency
 * @param item Item to compute saliency for
 * @param current_time Current timestamp
 * @return Attention saliency score
 */
float ce_workspace_compute_attention_saliency(const ce_item_t *item, double current_time);

/**
 * Compute novelty-based saliency
 * @param workspace Workspace instance
 * @param item Item to compute saliency for
 * @return Novelty saliency score
 */
float ce_workspace_compute_novelty_saliency(const ce_workspace_t *workspace, const ce_item_t *item);

/**
 * Compute goal-relevance saliency
 * @param workspace Workspace instance
 * @param item Item to compute saliency for
 * @return Goal relevance saliency score
 */
float ce_workspace_compute_goal_saliency(const ce_workspace_t *workspace, const ce_item_t *item);

/**
 * Compute uncertainty-based saliency
 * @param item Item to compute saliency for
 * @return Uncertainty saliency score
 */
float ce_workspace_compute_uncertainty_saliency(const ce_item_t *item);

/**
 * Perform attention arbitration
 * @param workspace Workspace instance
 * @param candidates Candidate items
 * @param selected Output array for selected items
 * @param max_selected Maximum number of items to select
 * @return Number of items selected
 */
size_t ce_workspace_arbitrate_attention(const ce_workspace_t *workspace,
                                       const ce_item_list_t *candidates,
                                       ce_item_t **selected, size_t max_selected);

/**
 * Broadcast selected items
 * @param workspace Workspace instance
 * @param items Items to broadcast
 * @param count Number of items
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_workspace_broadcast_items(ce_workspace_t *workspace, ce_item_t **items, size_t count);

/**
 * Clear expired goals
 * @param workspace Workspace instance
 * @return Number of goals cleared
 */
size_t ce_workspace_clear_expired_goals(ce_workspace_t *workspace);

#endif /* CE_WORKSPACE_H */
