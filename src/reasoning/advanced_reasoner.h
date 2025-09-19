/**
 * Consciousness Emulator v1.1 - Advanced Reasoning Engine
 * 
 * Implements sophisticated reasoning algorithms including forward/backward chaining,
 * probabilistic reasoning, causal inference, and analogical reasoning.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#ifndef CE_ADVANCED_REASONER_H
#define CE_ADVANCED_REASONER_H

#include "../../include/consciousness.h"
#include <stdbool.h>

/* ============================================================================
 * Reasoning Types and Modes
 * ============================================================================ */

typedef enum {
    CE_REASONING_MODE_FORWARD_CHAINING = 0,
    CE_REASONING_MODE_BACKWARD_CHAINING,
    CE_REASONING_MODE_PROBABILISTIC,
    CE_REASONING_MODE_CAUSAL,
    CE_REASONING_MODE_ANALOGICAL,
    CE_REASONING_MODE_ABDUCTIVE,
    CE_REASONING_MODE_DEDUCTIVE,
    CE_REASONING_MODE_INDUCTIVE,
    CE_REASONING_MODE_META_REASONING
} ce_reasoning_mode_t;

typedef enum {
    CE_RULE_TYPE_FACT = 0,
    CE_RULE_TYPE_IMPLICATION,
    CE_RULE_TYPE_CONSTRAINT,
    CE_RULE_TYPE_CAUSAL,
    CE_RULE_TYPE_ANALOGICAL,
    CE_RULE_TYPE_PROBABILISTIC,
    CE_RULE_TYPE_TEMPORAL,
    CE_RULE_TYPE_META
} ce_rule_type_t;

/* ============================================================================
 * Rule System
 * ============================================================================ */

typedef struct {
    uint64_t id;
    char name[64];
    ce_rule_type_t type;
    float confidence;
    float weight;
    double created_at;
    double last_used;
    uint64_t usage_count;
    
    /* Rule content */
    char *premise;          /* Premise condition */
    char *conclusion;       /* Conclusion */
    char *context;          /* Context information */
    
    /* Probabilistic information */
    float probability;
    float prior_probability;
    float likelihood;
    
    /* Causal information */
    float causal_strength;
    float temporal_delay;
    
    /* Analogical information */
    char *source_domain;
    char *target_domain;
    float analogy_strength;
    
    bool is_active;
} ce_reasoning_rule_t;

typedef struct {
    ce_reasoning_rule_t *rules;
    size_t count;
    size_t capacity;
    
    /* Rule indexing */
    struct {
        ce_reasoning_rule_t **by_type[8];  /* Indexed by rule type */
        size_t count_by_type[8];
    } index;
    
    /* Statistics */
    uint64_t total_rule_firings;
    double total_reasoning_time;
    double avg_reasoning_time;
} ce_rule_base_t;

/* ============================================================================
 * Reasoning Context
 * ============================================================================ */

typedef struct {
    ce_item_list_t *facts;          /* Current facts */
    ce_item_list_t *goals;          /* Current goals */
    ce_item_list_t *hypotheses;     /* Current hypotheses */
    ce_item_list_t *evidence;       /* Supporting evidence */
    
    /* Reasoning state */
    ce_reasoning_mode_t mode;
    float confidence_threshold;
    int max_inference_depth;
    int max_hypotheses;
    
    /* Context metadata */
    char *domain;                   /* Reasoning domain */
    char *context_description;      /* Context description */
    double timestamp;               /* Context timestamp */
    
    /* Reasoning history */
    struct {
        ce_item_t **inference_chain;
        size_t chain_length;
        size_t max_chain_length;
    } history;
} ce_reasoning_context_t;

/* ============================================================================
 * Advanced Reasoner Structure
 * ============================================================================ */

typedef struct ce_advanced_reasoner {
    ce_rule_base_t rule_base;
    
    /* Reasoning contexts */
    ce_reasoning_context_t *contexts;
    size_t context_count;
    size_t max_contexts;
    
    /* Configuration */
    ce_reasoning_mode_t default_mode;
    float default_confidence_threshold;
    int default_max_depth;
    bool enable_meta_reasoning;
    bool enable_uncertainty_propagation;
    
    /* Performance monitoring */
    uint64_t total_reasoning_cycles;
    uint64_t total_rule_firings;
    double total_reasoning_time;
    double avg_reasoning_time;
    double max_reasoning_time;
    
    /* Learning and adaptation */
    struct {
        bool enable_rule_learning;
        bool enable_confidence_adjustment;
        bool enable_rule_discovery;
        float learning_rate;
        uint64_t learning_cycles;
    } learning;
    
    /* Synchronization */
    pthread_mutex_t mutex;
    
    /* Callbacks */
    void (*on_rule_fired)(const ce_reasoning_rule_t *rule, const ce_item_t *conclusion, void *user_data);
    void (*on_inference_complete)(const ce_reasoning_context_t *context, void *user_data);
    void (*on_contradiction_detected)(const ce_item_t *item1, const ce_item_t *item2, void *user_data);
    void *callback_user_data;
} ce_advanced_reasoner_t;

/* ============================================================================
 * Advanced Reasoner API
 * ============================================================================ */

/**
 * Create advanced reasoner
 * @param default_mode Default reasoning mode
 * @param confidence_threshold Default confidence threshold
 * @param max_depth Default maximum inference depth
 * @return Advanced reasoner instance or NULL on error
 */
ce_advanced_reasoner_t *ce_advanced_reasoner_create(ce_reasoning_mode_t default_mode,
                                                    float confidence_threshold,
                                                    int max_depth);

/**
 * Free advanced reasoner
 * @param reasoner Advanced reasoner instance
 */
void ce_advanced_reasoner_free(ce_advanced_reasoner_t *reasoner);

/**
 * Add reasoning rule
 * @param reasoner Advanced reasoner instance
 * @param name Rule name
 * @param type Rule type
 * @param premise Rule premise
 * @param conclusion Rule conclusion
 * @param confidence Rule confidence
 * @return Rule ID or 0 on error
 */
uint64_t ce_advanced_reasoner_add_rule(ce_advanced_reasoner_t *reasoner,
                                       const char *name, ce_rule_type_t type,
                                       const char *premise, const char *conclusion,
                                       float confidence);

/**
 * Remove reasoning rule
 * @param reasoner Advanced reasoner instance
 * @param rule_id Rule ID to remove
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_advanced_reasoner_remove_rule(ce_advanced_reasoner_t *reasoner, uint64_t rule_id);

/**
 * Create reasoning context
 * @param reasoner Advanced reasoner instance
 * @param domain Reasoning domain
 * @param description Context description
 * @return Context ID or 0 on error
 */
uint64_t ce_advanced_reasoner_create_context(ce_advanced_reasoner_t *reasoner,
                                             const char *domain,
                                             const char *description);

/**
 * Add fact to context
 * @param reasoner Advanced reasoner instance
 * @param context_id Context ID
 * @param fact Fact to add
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_advanced_reasoner_add_fact(ce_advanced_reasoner_t *reasoner,
                                         uint64_t context_id,
                                         const ce_item_t *fact);

/**
 * Add goal to context
 * @param reasoner Advanced reasoner instance
 * @param context_id Context ID
 * @param goal Goal to add
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_advanced_reasoner_add_goal(ce_advanced_reasoner_t *reasoner,
                                         uint64_t context_id,
                                         const ce_item_t *goal);

/**
 * Perform reasoning in context
 * @param reasoner Advanced reasoner instance
 * @param context_id Context ID
 * @param mode Reasoning mode
 * @return Generated conclusions
 */
ce_item_list_t *ce_advanced_reasoner_reason(ce_advanced_reasoner_t *reasoner,
                                            uint64_t context_id,
                                            ce_reasoning_mode_t mode);

/**
 * Get reasoning statistics
 * @param reasoner Advanced reasoner instance
 * @param stats Output statistics structure
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_advanced_reasoner_get_stats(const ce_advanced_reasoner_t *reasoner, struct {
    size_t total_rules;
    size_t active_rules;
    size_t total_contexts;
    uint64_t total_reasoning_cycles;
    uint64_t total_rule_firings;
    double avg_reasoning_time;
    double max_reasoning_time;
    uint64_t learning_cycles;
} *stats);

/**
 * Set callback functions
 * @param reasoner Advanced reasoner instance
 * @param on_rule_fired Rule fired callback
 * @param on_inference_complete Inference complete callback
 * @param on_contradiction_detected Contradiction detected callback
 * @param user_data User data passed to callbacks
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_advanced_reasoner_set_callbacks(ce_advanced_reasoner_t *reasoner,
                                              void (*on_rule_fired)(const ce_reasoning_rule_t *rule, const ce_item_t *conclusion, void *user_data),
                                              void (*on_inference_complete)(const ce_reasoning_context_t *context, void *user_data),
                                              void (*on_contradiction_detected)(const ce_item_t *item1, const ce_item_t *item2, void *user_data),
                                              void *user_data);

/* ============================================================================
 * Specialized Reasoning Functions
 * ============================================================================ */

/**
 * Forward chaining reasoning
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @return Generated conclusions
 */
ce_item_list_t *ce_reasoning_forward_chaining(ce_advanced_reasoner_t *reasoner,
                                              ce_reasoning_context_t *context);

/**
 * Backward chaining reasoning
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @param goal Goal to achieve
 * @return Proof tree or NULL if goal cannot be achieved
 */
ce_item_list_t *ce_reasoning_backward_chaining(ce_advanced_reasoner_t *reasoner,
                                               ce_reasoning_context_t *context,
                                               const ce_item_t *goal);

/**
 * Probabilistic reasoning
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @param evidence Evidence items
 * @return Probabilistic conclusions
 */
ce_item_list_t *ce_reasoning_probabilistic(ce_advanced_reasoner_t *reasoner,
                                           ce_reasoning_context_t *context,
                                           const ce_item_list_t *evidence);

/**
 * Causal reasoning
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @param cause Causal item
 * @return Causal effects
 */
ce_item_list_t *ce_reasoning_causal(ce_advanced_reasoner_t *reasoner,
                                    ce_reasoning_context_t *context,
                                    const ce_item_t *cause);

/**
 * Analogical reasoning
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @param source_analogy Source analogy
 * @param target_domain Target domain
 * @return Analogical conclusions
 */
ce_item_list_t *ce_reasoning_analogical(ce_advanced_reasoner_t *reasoner,
                                        ce_reasoning_context_t *context,
                                        const ce_item_t *source_analogy,
                                        const char *target_domain);

/**
 * Abductive reasoning
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @param observation Observation to explain
 * @return Explanatory hypotheses
 */
ce_item_list_t *ce_reasoning_abductive(ce_advanced_reasoner_t *reasoner,
                                       ce_reasoning_context_t *context,
                                       const ce_item_t *observation);

/**
 * Meta-reasoning
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @return Meta-reasoning conclusions
 */
ce_item_list_t *ce_reasoning_meta(ce_advanced_reasoner_t *reasoner,
                                  ce_reasoning_context_t *context);

/* ============================================================================
 * Rule Management Functions
 * ============================================================================ */

/**
 * Parse rule from text
 * @param rule_text Rule text
 * @param rule Output rule structure
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_reasoning_parse_rule(const char *rule_text, ce_reasoning_rule_t *rule);

/**
 * Serialize rule to text
 * @param rule Rule to serialize
 * @param rule_text Output rule text
 * @param max_length Maximum text length
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_reasoning_serialize_rule(const ce_reasoning_rule_t *rule,
                                       char *rule_text, size_t max_length);

/**
 * Check rule consistency
 * @param reasoner Advanced reasoner instance
 * @param rule Rule to check
 * @return True if rule is consistent
 */
bool ce_reasoning_check_rule_consistency(ce_advanced_reasoner_t *reasoner,
                                         const ce_reasoning_rule_t *rule);

/**
 * Learn rule from examples
 * @param reasoner Advanced reasoner instance
 * @param examples Example facts
 * @param num_examples Number of examples
 * @return Learned rule or NULL on error
 */
ce_reasoning_rule_t *ce_reasoning_learn_rule(ce_advanced_reasoner_t *reasoner,
                                             const ce_item_t **examples,
                                             size_t num_examples);

/* ============================================================================
 * Uncertainty and Contradiction Handling
 * ============================================================================ */

/**
 * Detect contradictions
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @param contradictions Output contradiction pairs
 * @param max_contradictions Maximum contradictions to find
 * @return Number of contradictions found
 */
size_t ce_reasoning_detect_contradictions(ce_advanced_reasoner_t *reasoner,
                                          ce_reasoning_context_t *context,
                                          struct {
                                              ce_item_t *item1;
                                              ce_item_t *item2;
                                              float contradiction_strength;
                                          } *contradictions,
                                          size_t max_contradictions);

/**
 * Resolve contradictions
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @param contradictions Contradiction pairs
 * @param num_contradictions Number of contradictions
 * @return Resolution strategy
 */
ce_error_t ce_reasoning_resolve_contradictions(ce_advanced_reasoner_t *reasoner,
                                               ce_reasoning_context_t *context,
                                               const struct {
                                                   ce_item_t *item1;
                                                   ce_item_t *item2;
                                                   float contradiction_strength;
                                               } *contradictions,
                                               size_t num_contradictions);

/**
 * Propagate uncertainty
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @param uncertainty_sources Sources of uncertainty
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_reasoning_propagate_uncertainty(ce_advanced_reasoner_t *reasoner,
                                              ce_reasoning_context_t *context,
                                              const ce_item_list_t *uncertainty_sources);

/* ============================================================================
 * Explanation Generation
 * ============================================================================ */

/**
 * Generate explanation for conclusion
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @param conclusion Conclusion to explain
 * @return Explanation text (caller must free)
 */
char *ce_reasoning_generate_explanation(ce_advanced_reasoner_t *reasoner,
                                        ce_reasoning_context_t *context,
                                        const ce_item_t *conclusion);

/**
 * Generate reasoning trace
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @return Reasoning trace (caller must free)
 */
char *ce_reasoning_generate_trace(ce_advanced_reasoner_t *reasoner,
                                  ce_reasoning_context_t *context);

/**
 * Generate confidence assessment
 * @param reasoner Advanced reasoner instance
 * @param context Reasoning context
 * @param conclusion Conclusion to assess
 * @return Confidence assessment
 */
float ce_reasoning_assess_confidence(ce_advanced_reasoner_t *reasoner,
                                     ce_reasoning_context_t *context,
                                     const ce_item_t *conclusion);

#endif /* CE_ADVANCED_REASONER_H */
