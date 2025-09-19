/**
 * Consciousness Emulator - Global Workspace Theory Implementation
 * 
 * Implements the central attention and arbitration mechanism based on
 * Global Workspace Theory principles.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "workspace.h"
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
 * Initialize goal system
 */
static ce_error_t init_goal_system(ce_workspace_t *workspace) {
    workspace->goal_system.goals = calloc(16, sizeof(ce_goal_t));
    if (!workspace->goal_system.goals) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    workspace->goal_system.count = 0;
    workspace->goal_system.capacity = 16;
    workspace->goal_system.goal_vector_dim = 64; /* Default dimension */
    
    workspace->goal_system.current_goal_vector = calloc(workspace->goal_system.goal_vector_dim, sizeof(float));
    if (!workspace->goal_system.current_goal_vector) {
        free(workspace->goal_system.goals);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    return CE_SUCCESS;
}

/**
 * Cleanup goal system
 */
static void cleanup_goal_system(ce_workspace_t *workspace) {
    if (workspace->goal_system.goals) {
        for (size_t i = 0; i < workspace->goal_system.count; i++) {
            ce_goal_t *goal = &workspace->goal_system.goals[i];
            if (goal->description) {
                free(goal->description);
            }
            if (goal->embedding) {
                free(goal->embedding);
            }
        }
        free(workspace->goal_system.goals);
    }
    
    if (workspace->goal_system.current_goal_vector) {
        free(workspace->goal_system.current_goal_vector);
    }
}

/**
 * Compute attention-based saliency (recency and access frequency)
 */
static float compute_attention_saliency(const ce_item_t *item, double current_time) {
    if (!item) {
        return 0.0f;
    }
    
    /* Recency component */
    double time_since_creation = current_time - item->timestamp;
    double time_since_access = current_time - item->last_accessed;
    
    float recency_score = (float)exp(-time_since_access * 0.1); /* Decay factor */
    
    /* Access frequency component (simplified) */
    float access_score = 1.0f; /* Could be enhanced with actual access tracking */
    
    return (recency_score + access_score) / 2.0f;
}

/**
 * Compute novelty-based saliency
 */
static float compute_novelty_saliency(const ce_workspace_t *workspace, const ce_item_t *item) {
    if (!item || !item->embedding || item->embedding_dim == 0) {
        return 0.5f; /* Default novelty for items without embeddings */
    }
    
    const ce_item_list_t *wm_items = ce_wm_get_items(workspace->wm);
    if (!wm_items || wm_items->count == 0) {
        return 1.0f; /* Maximum novelty for first item */
    }
    
    float min_similarity = 1.0f;
    size_t comparison_count = 0;
    
    /* Compare with other items in working memory */
    for (size_t i = 0; i < wm_items->count; i++) {
        const ce_item_t *other_item = wm_items->items[i];
        
        if (other_item != item && other_item->embedding && 
            other_item->embedding_dim == item->embedding_dim) {
            
            float similarity = ce_cosine_similarity(item->embedding, other_item->embedding, 
                                                  item->embedding_dim);
            similarity = fabsf(similarity); /* Use absolute similarity */
            
            if (similarity < min_similarity) {
                min_similarity = similarity;
            }
            comparison_count++;
        }
    }
    
    /* Novelty is inverse of maximum similarity */
    float novelty = comparison_count > 0 ? 1.0f - min_similarity : 1.0f;
    return fmaxf(0.0f, fminf(1.0f, novelty));
}

/**
 * Compute goal-relevance saliency
 */
static float compute_goal_saliency(const ce_workspace_t *workspace, const ce_item_t *item) {
    if (!item || !workspace->goal_system.current_goal_vector) {
        return 0.0f;
    }
    
    /* If item has no embedding, use default relevance */
    if (!item->embedding || item->embedding_dim == 0) {
        return 0.3f;
    }
    
    /* Compute similarity with current goal vector */
    if (item->embedding_dim == workspace->goal_system.goal_vector_dim) {
        float similarity = ce_cosine_similarity(item->embedding, 
                                              workspace->goal_system.current_goal_vector,
                                              item->embedding_dim);
        return fmaxf(0.0f, similarity);
    }
    
    return 0.0f;
}

/**
 * Compute uncertainty-based saliency
 */
static float compute_uncertainty_saliency(const ce_item_t *item) {
    if (!item) {
        return 0.0f;
    }
    
    /* Uncertainty is inverse of confidence */
    float uncertainty = 1.0f - item->confidence;
    
    /* Boost uncertainty for questions and predictions */
    if (item->type == CE_ITEM_TYPE_QUESTION || item->type == CE_ITEM_TYPE_PREDICTION) {
        uncertainty *= 1.5f;
    }
    
    return fminf(1.0f, uncertainty);
}

/**
 * Perform winner-take-all arbitration
 */
static size_t arbitrate_winner_take_all(const ce_workspace_t *workspace,
                                       const ce_item_list_t *candidates,
                                       ce_item_t **selected, size_t max_selected) {
    if (!candidates || candidates->count == 0 || max_selected == 0) {
        return 0;
    }
    
    /* Find item with highest saliency */
    ce_item_t *best_item = NULL;
    float best_saliency = -1.0f;
    
    for (size_t i = 0; i < candidates->count; i++) {
        ce_item_t *item = candidates->items[i];
        if (item->saliency > best_saliency) {
            best_saliency = item->saliency;
            best_item = item;
        }
    }
    
    if (best_item && best_saliency >= workspace->threshold) {
        selected[0] = best_item;
        return 1;
    }
    
    return 0;
}

/**
 * Perform top-k arbitration
 */
static size_t arbitrate_top_k(const ce_workspace_t *workspace,
                             const ce_item_list_t *candidates,
                             ce_item_t **selected, size_t max_selected) {
    if (!candidates || candidates->count == 0 || max_selected == 0) {
        return 0;
    }
    
    /* Create array of items with saliencies */
    struct {
        ce_item_t *item;
        float saliency;
    } *item_saliencies = malloc(candidates->count * sizeof(*item_saliencies));
    
    if (!item_saliencies) {
        return 0;
    }
    
    /* Copy items and saliencies */
    for (size_t i = 0; i < candidates->count; i++) {
        item_saliencies[i].item = candidates->items[i];
        item_saliencies[i].saliency = candidates->items[i]->saliency;
    }
    
    /* Sort by saliency (simple bubble sort) */
    for (size_t i = 0; i < candidates->count - 1; i++) {
        for (size_t j = 0; j < candidates->count - i - 1; j++) {
            if (item_saliencies[j].saliency < item_saliencies[j + 1].saliency) {
                struct { ce_item_t *item; float saliency; } temp = item_saliencies[j];
                item_saliencies[j] = item_saliencies[j + 1];
                item_saliencies[j + 1] = temp;
            }
        }
    }
    
    /* Select top items above threshold */
    size_t selected_count = 0;
    for (size_t i = 0; i < candidates->count && selected_count < max_selected; i++) {
        if (item_saliencies[i].saliency >= workspace->threshold) {
            selected[selected_count] = item_saliencies[i].item;
            selected_count++;
        }
    }
    
    free(item_saliencies);
    return selected_count;
}

/**
 * Perform threshold-based arbitration
 */
static size_t arbitrate_threshold(const ce_workspace_t *workspace,
                                 const ce_item_list_t *candidates,
                                 ce_item_t **selected, size_t max_selected) {
    if (!candidates || candidates->count == 0 || max_selected == 0) {
        return 0;
    }
    
    size_t selected_count = 0;
    
    for (size_t i = 0; i < candidates->count && selected_count < max_selected; i++) {
        ce_item_t *item = candidates->items[i];
        if (item->saliency >= workspace->threshold) {
            selected[selected_count] = item;
            selected_count++;
        }
    }
    
    return selected_count;
}

/* ============================================================================
 * Public API Implementation
 * ============================================================================ */

ce_workspace_t *ce_workspace_create(ce_working_memory_t *wm, float threshold) {
    ce_saliency_weights_t default_weights = {
        .attention_weight = CE_WORKSPACE_DEFAULT_ATTENTION_WEIGHT,
        .novelty_weight = CE_WORKSPACE_DEFAULT_NOVELTY_WEIGHT,
        .goal_weight = CE_WORKSPACE_DEFAULT_GOAL_WEIGHT,
        .uncertainty_weight = CE_WORKSPACE_DEFAULT_UNCERTAINTY_WEIGHT,
        .recency_weight = 0.0f
    };
    
    return ce_workspace_create_with_config(wm, threshold, CE_ATTENTION_MODE_TOP_K,
                                          CE_WORKSPACE_MAX_BROADCAST_ITEMS, &default_weights);
}

ce_workspace_t *ce_workspace_create_with_config(ce_working_memory_t *wm, float threshold,
                                               ce_attention_mode_t attention_mode,
                                               size_t max_broadcast_items,
                                               const ce_saliency_weights_t *saliency_weights) {
    if (!wm) {
        return NULL;
    }
    
    ce_workspace_t *workspace = calloc(1, sizeof(ce_workspace_t));
    if (!workspace) {
        return NULL;
    }
    
    workspace->wm = wm;
    workspace->threshold = threshold;
    workspace->attention_mode = attention_mode;
    workspace->max_broadcast_items = max_broadcast_items;
    
    if (saliency_weights) {
        workspace->saliency_weights = *saliency_weights;
    } else {
        workspace->saliency_weights.attention_weight = CE_WORKSPACE_DEFAULT_ATTENTION_WEIGHT;
        workspace->saliency_weights.novelty_weight = CE_WORKSPACE_DEFAULT_NOVELTY_WEIGHT;
        workspace->saliency_weights.goal_weight = CE_WORKSPACE_DEFAULT_GOAL_WEIGHT;
        workspace->saliency_weights.uncertainty_weight = CE_WORKSPACE_DEFAULT_UNCERTAINTY_WEIGHT;
        workspace->saliency_weights.recency_weight = 0.0f;
    }
    
    /* Initialize broadcast items list */
    workspace->broadcast_items = ce_item_list_create(max_broadcast_items);
    if (!workspace->broadcast_items) {
        free(workspace);
        return NULL;
    }
    
    /* Initialize goal system */
    if (init_goal_system(workspace) != CE_SUCCESS) {
        ce_item_list_free(workspace->broadcast_items);
        free(workspace);
        return NULL;
    }
    
    /* Initialize statistics */
    workspace->total_broadcasts = 0;
    workspace->total_arbitrations = 0;
    workspace->total_arbitration_time = 0.0;
    workspace->avg_arbitration_time = 0.0;
    workspace->max_arbitration_time = 0.0;
    workspace->current_broadcast_id = 0;
    workspace->last_broadcast_time = 0.0;
    
    /* Initialize callbacks */
    workspace->on_broadcast = NULL;
    workspace->on_arbitration = NULL;
    workspace->callback_user_data = NULL;
    
    /* Initialize synchronization */
    if (pthread_mutex_init(&workspace->mutex, NULL) != 0) {
        cleanup_goal_system(workspace);
        ce_item_list_free(workspace->broadcast_items);
        free(workspace);
        return NULL;
    }
    
    return workspace;
}

void ce_workspace_free(ce_workspace_t *workspace) {
    if (!workspace) {
        return;
    }
    
    /* Cleanup goal system */
    cleanup_goal_system(workspace);
    
    /* Free broadcast items list */
    if (workspace->broadcast_items) {
        ce_item_list_free(workspace->broadcast_items);
    }
    
    /* Destroy synchronization */
    pthread_mutex_destroy(&workspace->mutex);
    
    /* Free workspace */
    free(workspace);
}

ce_error_t ce_workspace_process(ce_workspace_t *workspace) {
    if (!workspace) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&workspace->mutex);
    
    double arbitration_start = ce_get_timestamp();
    
    /* Get working memory items */
    const ce_item_list_t *wm_items = ce_wm_get_items(workspace->wm);
    if (!wm_items || wm_items->count == 0) {
        pthread_mutex_unlock(&workspace->mutex);
        return CE_SUCCESS;
    }
    
    /* Compute enhanced saliency for all items */
    double current_time = ce_get_timestamp();
    for (size_t i = 0; i < wm_items->count; i++) {
        ce_item_t *item = wm_items->items[i];
        float enhanced_saliency = ce_workspace_compute_saliency(workspace, item, current_time);
        ce_item_update_saliency(item, enhanced_saliency);
    }
    
    /* Perform attention arbitration */
    ce_item_t *selected_items[CE_WORKSPACE_MAX_BROADCAST_ITEMS];
    size_t selected_count = ce_workspace_arbitrate_attention(workspace, wm_items, 
                                                           selected_items, 
                                                           workspace->max_broadcast_items);
    
    /* Broadcast selected items */
    if (selected_count > 0) {
        ce_workspace_broadcast_items(workspace, selected_items, selected_count);
    }
    
    /* Update statistics */
    double arbitration_end = ce_get_timestamp();
    double arbitration_time = arbitration_end - arbitration_start;
    
    workspace->total_arbitrations++;
    workspace->total_arbitration_time += arbitration_time;
    workspace->avg_arbitration_time = workspace->total_arbitration_time / workspace->total_arbitrations;
    
    if (arbitration_time > workspace->max_arbitration_time) {
        workspace->max_arbitration_time = arbitration_time;
    }
    
    pthread_mutex_unlock(&workspace->mutex);
    
    return CE_SUCCESS;
}

const ce_item_list_t *ce_workspace_get_broadcast(const ce_workspace_t *workspace) {
    if (!workspace) {
        return NULL;
    }
    
    return workspace->broadcast_items;
}

ce_error_t ce_workspace_set_attention_mode(ce_workspace_t *workspace, ce_attention_mode_t mode) {
    if (!workspace) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&workspace->mutex);
    workspace->attention_mode = mode;
    pthread_mutex_unlock(&workspace->mutex);
    
    return CE_SUCCESS;
}

ce_error_t ce_workspace_set_saliency_weights(ce_workspace_t *workspace, 
                                            const ce_saliency_weights_t *weights) {
    if (!workspace || !weights) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&workspace->mutex);
    workspace->saliency_weights = *weights;
    pthread_mutex_unlock(&workspace->mutex);
    
    return CE_SUCCESS;
}

ce_error_t ce_workspace_get_stats(const ce_workspace_t *workspace, struct {
    uint64_t total_broadcasts;
    uint64_t total_arbitrations;
    double avg_arbitration_time;
    double max_arbitration_time;
    size_t current_broadcast_count;
    size_t active_goals;
} *stats) {
    if (!workspace || !stats) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock((pthread_mutex_t *)&workspace->mutex);
    
    stats->total_broadcasts = workspace->total_broadcasts;
    stats->total_arbitrations = workspace->total_arbitrations;
    stats->avg_arbitration_time = workspace->avg_arbitration_time;
    stats->max_arbitration_time = workspace->max_arbitration_time;
    stats->current_broadcast_count = workspace->broadcast_items->count;
    
    /* Count active goals */
    stats->active_goals = 0;
    for (size_t i = 0; i < workspace->goal_system.count; i++) {
        if (workspace->goal_system.goals[i].is_active) {
            stats->active_goals++;
        }
    }
    
    pthread_mutex_unlock((pthread_mutex_t *)&workspace->mutex);
    
    return CE_SUCCESS;
}

ce_error_t ce_workspace_set_callbacks(ce_workspace_t *workspace,
                                     void (*on_broadcast)(const ce_item_t *item, void *user_data),
                                     void (*on_arbitration)(const ce_item_list_t *selected_items, void *user_data),
                                     void *user_data) {
    if (!workspace) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&workspace->mutex);
    
    workspace->on_broadcast = on_broadcast;
    workspace->on_arbitration = on_arbitration;
    workspace->callback_user_data = user_data;
    
    pthread_mutex_unlock(&workspace->mutex);
    
    return CE_SUCCESS;
}

/* ============================================================================
 * Goal System Implementation
 * ============================================================================ */

uint64_t ce_workspace_add_goal(ce_workspace_t *workspace, const char *description,
                              float priority, const float *embedding, size_t embedding_dim,
                              double deadline) {
    if (!workspace || !description) {
        return 0;
    }
    
    pthread_mutex_lock(&workspace->mutex);
    
    /* Resize goals array if needed */
    if (workspace->goal_system.count >= workspace->goal_system.capacity) {
        size_t new_capacity = workspace->goal_system.capacity * 2;
        ce_goal_t *new_goals = realloc(workspace->goal_system.goals, 
                                      new_capacity * sizeof(ce_goal_t));
        if (!new_goals) {
            pthread_mutex_unlock(&workspace->mutex);
            return 0;
        }
        
        workspace->goal_system.goals = new_goals;
        workspace->goal_system.capacity = new_capacity;
    }
    
    /* Add new goal */
    ce_goal_t *goal = &workspace->goal_system.goals[workspace->goal_system.count];
    
    static uint64_t goal_id_counter = 1;
    goal->id = goal_id_counter++;
    goal->description = malloc(strlen(description) + 1);
    if (!goal->description) {
        pthread_mutex_unlock(&workspace->mutex);
        return 0;
    }
    strcpy(goal->description, description);
    
    goal->priority = priority;
    goal->created_at = ce_get_timestamp();
    goal->deadline = deadline;
    goal->is_active = true;
    
    /* Copy embedding if provided */
    if (embedding && embedding_dim > 0) {
        goal->embedding = malloc(embedding_dim * sizeof(float));
        if (!goal->embedding) {
            free(goal->description);
            pthread_mutex_unlock(&workspace->mutex);
            return 0;
        }
        memcpy(goal->embedding, embedding, embedding_dim * sizeof(float));
        goal->embedding_dim = embedding_dim;
    } else {
        goal->embedding = NULL;
        goal->embedding_dim = 0;
    }
    
    workspace->goal_system.count++;
    
    /* Update goal vector */
    ce_workspace_update_goal_vector(workspace);
    
    pthread_mutex_unlock(&workspace->mutex);
    
    return goal->id;
}

ce_error_t ce_workspace_remove_goal(ce_workspace_t *workspace, uint64_t goal_id) {
    if (!workspace) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&workspace->mutex);
    
    for (size_t i = 0; i < workspace->goal_system.count; i++) {
        ce_goal_t *goal = &workspace->goal_system.goals[i];
        
        if (goal->id == goal_id) {
            /* Free goal resources */
            if (goal->description) {
                free(goal->description);
            }
            if (goal->embedding) {
                free(goal->embedding);
            }
            
            /* Shift remaining goals */
            for (size_t j = i; j < workspace->goal_system.count - 1; j++) {
                workspace->goal_system.goals[j] = workspace->goal_system.goals[j + 1];
            }
            
            workspace->goal_system.count--;
            
            /* Update goal vector */
            ce_workspace_update_goal_vector(workspace);
            
            pthread_mutex_unlock(&workspace->mutex);
            return CE_SUCCESS;
        }
    }
    
    pthread_mutex_unlock(&workspace->mutex);
    return CE_ERROR_UNKNOWN; /* Goal not found */
}

ce_error_t ce_workspace_update_goal_priority(ce_workspace_t *workspace, uint64_t goal_id, float priority) {
    if (!workspace) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&workspace->mutex);
    
    for (size_t i = 0; i < workspace->goal_system.count; i++) {
        ce_goal_t *goal = &workspace->goal_system.goals[i];
        
        if (goal->id == goal_id) {
            goal->priority = priority;
            ce_workspace_update_goal_vector(workspace);
            pthread_mutex_unlock(&workspace->mutex);
            return CE_SUCCESS;
        }
    }
    
    pthread_mutex_unlock(&workspace->mutex);
    return CE_ERROR_UNKNOWN; /* Goal not found */
}

size_t ce_workspace_get_active_goals(const ce_workspace_t *workspace, ce_goal_t **goals, size_t max_goals) {
    if (!workspace || !goals || max_goals == 0) {
        return 0;
    }
    
    pthread_mutex_lock((pthread_mutex_t *)&workspace->mutex);
    
    size_t count = 0;
    for (size_t i = 0; i < workspace->goal_system.count && count < max_goals; i++) {
        const ce_goal_t *goal = &workspace->goal_system.goals[i];
        if (goal->is_active) {
            goals[count] = (ce_goal_t *)goal;
            count++;
        }
    }
    
    pthread_mutex_unlock((pthread_mutex_t *)&workspace->mutex);
    
    return count;
}

ce_error_t ce_workspace_update_goal_vector(ce_workspace_t *workspace) {
    if (!workspace) {
        return CE_ERROR_NULL_POINTER;
    }
    
    /* Initialize goal vector to zero */
    memset(workspace->goal_system.current_goal_vector, 0, 
           workspace->goal_system.goal_vector_dim * sizeof(float));
    
    if (workspace->goal_system.count == 0) {
        return CE_SUCCESS;
    }
    
    /* Compute weighted average of active goal embeddings */
    float total_weight = 0.0f;
    size_t valid_goals = 0;
    
    for (size_t i = 0; i < workspace->goal_system.count; i++) {
        const ce_goal_t *goal = &workspace->goal_system.goals[i];
        
        if (goal->is_active && goal->embedding && 
            goal->embedding_dim == workspace->goal_system.goal_vector_dim) {
            
            for (size_t j = 0; j < workspace->goal_system.goal_vector_dim; j++) {
                workspace->goal_system.current_goal_vector[j] += goal->embedding[j] * goal->priority;
            }
            
            total_weight += goal->priority;
            valid_goals++;
        }
    }
    
    /* Normalize by total weight */
    if (total_weight > 0.0f) {
        for (size_t j = 0; j < workspace->goal_system.goal_vector_dim; j++) {
            workspace->goal_system.current_goal_vector[j] /= total_weight;
        }
    }
    
    return CE_SUCCESS;
}

/* ============================================================================
 * Internal Function Implementations
 * ============================================================================ */

float ce_workspace_compute_saliency(const ce_workspace_t *workspace, const ce_item_t *item, double current_time) {
    if (!workspace || !item) {
        return 0.0f;
    }
    
    const ce_saliency_weights_t *weights = &workspace->saliency_weights;
    
    /* Compute individual saliency components */
    float attention_saliency = compute_attention_saliency(item, current_time);
    float novelty_saliency = compute_novelty_saliency(workspace, item);
    float goal_saliency = compute_goal_saliency(workspace, item);
    float uncertainty_saliency = compute_uncertainty_saliency(item);
    
    /* Weighted combination */
    float total_saliency = 
        weights->attention_weight * attention_saliency +
        weights->novelty_weight * novelty_saliency +
        weights->goal_weight * goal_saliency +
        weights->uncertainty_weight * uncertainty_saliency;
    
    /* Normalize to [0, 1] range */
    return fmaxf(0.0f, fminf(1.0f, total_saliency));
}

float ce_workspace_compute_attention_saliency(const ce_item_t *item, double current_time) {
    return compute_attention_saliency(item, current_time);
}

float ce_workspace_compute_novelty_saliency(const ce_workspace_t *workspace, const ce_item_t *item) {
    return compute_novelty_saliency(workspace, item);
}

float ce_workspace_compute_goal_saliency(const ce_workspace_t *workspace, const ce_item_t *item) {
    return compute_goal_saliency(workspace, item);
}

float ce_workspace_compute_uncertainty_saliency(const ce_item_t *item) {
    return compute_uncertainty_saliency(item);
}

size_t ce_workspace_arbitrate_attention(const ce_workspace_t *workspace,
                                       const ce_item_list_t *candidates,
                                       ce_item_t **selected, size_t max_selected) {
    if (!workspace || !candidates || !selected || max_selected == 0) {
        return 0;
    }
    
    switch (workspace->attention_mode) {
        case CE_ATTENTION_MODE_WINNER_TAKE_ALL:
            return arbitrate_winner_take_all(workspace, candidates, selected, max_selected);
            
        case CE_ATTENTION_MODE_TOP_K:
            return arbitrate_top_k(workspace, candidates, selected, max_selected);
            
        case CE_ATTENTION_MODE_THRESHOLD:
            return arbitrate_threshold(workspace, candidates, selected, max_selected);
            
        case CE_ATTENTION_MODE_WEIGHTED:
            /* For now, use top-k as weighted mode */
            return arbitrate_top_k(workspace, candidates, selected, max_selected);
            
        default:
            return 0;
    }
}

ce_error_t ce_workspace_broadcast_items(ce_workspace_t *workspace, ce_item_t **items, size_t count) {
    if (!workspace || !items || count == 0) {
        return CE_ERROR_NULL_POINTER;
    }
    
    /* Clear previous broadcast */
    ce_item_list_free(workspace->broadcast_items);
    workspace->broadcast_items = ce_item_list_create(workspace->max_broadcast_items);
    
    if (!workspace->broadcast_items) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Add items to broadcast list */
    for (size_t i = 0; i < count; i++) {
        ce_item_list_add(workspace->broadcast_items, items[i]);
        
        /* Call broadcast callback */
        if (workspace->on_broadcast) {
            workspace->on_broadcast(items[i], workspace->callback_user_data);
        }
    }
    
    /* Update broadcast statistics */
    workspace->total_broadcasts += count;
    workspace->current_broadcast_id++;
    workspace->last_broadcast_time = ce_get_timestamp();
    
    /* Call arbitration callback */
    if (workspace->on_arbitration) {
        workspace->on_arbitration(workspace->broadcast_items, workspace->callback_user_data);
    }
    
    return CE_SUCCESS;
}

size_t ce_workspace_clear_expired_goals(ce_workspace_t *workspace) {
    if (!workspace) {
        return 0;
    }
    
    pthread_mutex_lock(&workspace->mutex);
    
    double current_time = ce_get_timestamp();
    size_t cleared_count = 0;
    
    for (size_t i = 0; i < workspace->goal_system.count; i++) {
        ce_goal_t *goal = &workspace->goal_system.goals[i];
        
        if (goal->is_active && goal->deadline > 0.0 && current_time > goal->deadline) {
            goal->is_active = false;
            cleared_count++;
        }
    }
    
    if (cleared_count > 0) {
        ce_workspace_update_goal_vector(workspace);
    }
    
    pthread_mutex_unlock(&workspace->mutex);
    
    return cleared_count;
}
