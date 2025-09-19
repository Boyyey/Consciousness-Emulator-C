/**
 * Consciousness Emulator v1.1 - Web Visualization Interface
 * 
 * Real-time web-based visualization and monitoring interface for the CE system.
 * Provides interactive dashboards, cognitive state visualization, and control panels.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#ifndef CE_WEB_INTERFACE_H
#define CE_WEB_INTERFACE_H

#include "../../include/consciousness.h"
#include <stdbool.h>

/* ============================================================================
 * Web Interface Configuration
 * ============================================================================ */

#define CE_WEB_DEFAULT_PORT 8080
#define CE_WEB_MAX_CONNECTIONS 100
#define CE_WEB_MAX_MESSAGE_SIZE 8192
#define CE_WEB_UPDATE_INTERVAL 0.1  /* 100ms */
#define CE_WEB_MAX_HISTORY 1000

/* ============================================================================
 * WebSocket Message Types
 * ============================================================================ */

typedef enum {
    CE_WEB_MSG_TYPE_SYSTEM_STATUS = 0,
    CE_WEB_MSG_TYPE_WORKING_MEMORY,
    CE_WEB_MSG_TYPE_BROADCAST,
    CE_WEB_MSG_TYPE_LTM_STATS,
    CE_WEB_MSG_TYPE_REASONING_TRACE,
    CE_WEB_MSG_TYPE_NEURAL_STATS,
    CE_WEB_MSG_TYPE_CUDA_STATS,
    CE_WEB_MSG_TYPE_USER_COMMAND,
    CE_WEB_MSG_TYPE_ERROR,
    CE_WEB_MSG_TYPE_HEARTBEAT
} ce_web_message_type_t;

typedef struct {
    ce_web_message_type_t type;
    char *data;                    /* JSON data */
    size_t data_size;
    double timestamp;
    uint64_t sequence_id;
} ce_web_message_t;

/* ============================================================================
 * Web Client Management
 * ============================================================================ */

typedef struct {
    int socket_fd;
    char client_id[64];
    char ip_address[16];
    double connected_at;
    double last_activity;
    bool is_authenticated;
    bool is_subscribed;
    
    /* Subscription preferences */
    struct {
        bool system_status;
        bool working_memory;
        bool broadcasts;
        bool ltm_stats;
        bool reasoning_trace;
        bool neural_stats;
        bool cuda_stats;
    } subscriptions;
    
    /* Message queue */
    ce_web_message_t *message_queue;
    size_t queue_size;
    size_t queue_capacity;
} ce_web_client_t;

typedef struct {
    ce_web_client_t *clients;
    size_t count;
    size_t capacity;
    pthread_mutex_t mutex;
} ce_web_client_pool_t;

/* ============================================================================
 * Visualization Data Structures
 * ============================================================================ */

typedef struct {
    /* System status */
    struct {
        double uptime;
        uint64_t total_ticks;
        double avg_tick_time;
        double max_tick_time;
        size_t active_modules;
        size_t message_queue_size;
    } system;
    
    /* Working memory */
    struct {
        size_t capacity;
        size_t count;
        double total_saliency;
        double avg_saliency;
        struct {
            uint64_t id;
            char content[256];
            float saliency;
            float confidence;
            double timestamp;
        } items[32];
    } working_memory;
    
    /* Global workspace */
    struct {
        size_t broadcast_count;
        struct {
            uint64_t id;
            char content[256];
            float saliency;
            double timestamp;
        } broadcasts[16];
    } workspace;
    
    /* Long-term memory */
    struct {
        size_t total_episodes;
        size_t consolidated_episodes;
        size_t semantic_index_size;
        uint64_t total_searches;
        double avg_search_time;
    } ltm;
    
    /* Neural engine */
    struct {
        size_t loaded_models;
        uint64_t total_inferences;
        double avg_inference_time;
        double max_inference_time;
    } neural;
    
    /* CUDA acceleration */
    struct {
        bool available;
        int device_count;
        int current_device;
        size_t total_memory;
        size_t free_memory;
        double gpu_utilization;
    } cuda;
    
    /* Advanced reasoning */
    struct {
        size_t total_rules;
        size_t active_rules;
        uint64_t total_reasoning_cycles;
        uint64_t total_rule_firings;
        double avg_reasoning_time;
    } reasoning;
} ce_web_visualization_data_t;

/* ============================================================================
 * Web Interface Structure
 * ============================================================================ */

typedef struct ce_web_interface {
    /* Server configuration */
    int server_port;
    int server_socket;
    bool is_running;
    
    /* Client management */
    ce_web_client_pool_t client_pool;
    
    /* Visualization data */
    ce_web_visualization_data_t viz_data;
    pthread_mutex_t data_mutex;
    
    /* Update thread */
    pthread_t update_thread;
    bool update_thread_running;
    double update_interval;
    
    /* History tracking */
    struct {
        ce_web_visualization_data_t *history;
        size_t count;
        size_t capacity;
        size_t current_index;
    } history;
    
    /* Statistics */
    uint64_t total_connections;
    uint64_t total_messages_sent;
    uint64_t total_bytes_sent;
    double total_uptime;
    
    /* Callbacks */
    void (*on_client_connected)(const char *client_id, void *user_data);
    void (*on_client_disconnected)(const char *client_id, void *user_data);
    void (*on_user_command)(const char *client_id, const char *command, void *user_data);
    void *callback_user_data;
} ce_web_interface_t;

/* ============================================================================
 * Web Interface API
 * ============================================================================ */

/**
 * Create web interface
 * @param port Server port
 * @param update_interval Update interval in seconds
 * @return Web interface instance or NULL on error
 */
ce_web_interface_t *ce_web_interface_create(int port, double update_interval);

/**
 * Free web interface
 * @param web_interface Web interface instance
 */
void ce_web_interface_free(ce_web_interface_t *web_interface);

/**
 * Start web server
 * @param web_interface Web interface instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_interface_start(ce_web_interface_t *web_interface);

/**
 * Stop web server
 * @param web_interface Web interface instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_interface_stop(ce_web_interface_t *web_interface);

/**
 * Update visualization data
 * @param web_interface Web interface instance
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_interface_update_data(ce_web_interface_t *web_interface);

/**
 * Send message to client
 * @param web_interface Web interface instance
 * @param client_id Client ID
 * @param message Message to send
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_interface_send_message(ce_web_interface_t *web_interface,
                                         const char *client_id,
                                         const ce_web_message_t *message);

/**
 * Broadcast message to all clients
 * @param web_interface Web interface instance
 * @param message Message to broadcast
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_interface_broadcast_message(ce_web_interface_t *web_interface,
                                              const ce_web_message_t *message);

/**
 * Get web interface statistics
 * @param web_interface Web interface instance
 * @param stats Output statistics structure
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_interface_get_stats(const ce_web_interface_t *web_interface, struct {
    int server_port;
    bool is_running;
    size_t connected_clients;
    uint64_t total_connections;
    uint64_t total_messages_sent;
    uint64_t total_bytes_sent;
    double total_uptime;
} *stats);

/**
 * Set callback functions
 * @param web_interface Web interface instance
 * @param on_client_connected Client connected callback
 * @param on_client_disconnected Client disconnected callback
 * @param on_user_command User command callback
 * @param user_data User data passed to callbacks
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_interface_set_callbacks(ce_web_interface_t *web_interface,
                                          void (*on_client_connected)(const char *client_id, void *user_data),
                                          void (*on_client_disconnected)(const char *client_id, void *user_data),
                                          void (*on_user_command)(const char *client_id, const char *command, void *user_data),
                                          void *user_data);

/* ============================================================================
 * Web Message Management
 * ============================================================================ */

/**
 * Create web message
 * @param type Message type
 * @param data JSON data
 * @param data_size Data size
 * @return Web message or NULL on error
 */
ce_web_message_t *ce_web_message_create(ce_web_message_type_t type,
                                        const char *data, size_t data_size);

/**
 * Free web message
 * @param message Web message
 */
void ce_web_message_free(ce_web_message_t *message);

/**
 * Serialize web message to JSON
 * @param message Web message
 * @param json_output Output JSON string
 * @param max_size Maximum output size
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_message_serialize(const ce_web_message_t *message,
                                    char *json_output, size_t max_size);

/**
 * Deserialize web message from JSON
 * @param json_input Input JSON string
 * @param message Output web message
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_message_deserialize(const char *json_input,
                                      ce_web_message_t *message);

/* ============================================================================
 * Visualization Data Functions
 * ============================================================================ */

/**
 * Create system status message
 * @param web_interface Web interface instance
 * @return System status message
 */
ce_web_message_t *ce_web_create_system_status_message(ce_web_interface_t *web_interface);

/**
 * Create working memory message
 * @param web_interface Web interface instance
 * @return Working memory message
 */
ce_web_message_t *ce_web_create_working_memory_message(ce_web_interface_t *web_interface);

/**
 * Create broadcast message
 * @param web_interface Web interface instance
 * @return Broadcast message
 */
ce_web_message_t *ce_web_create_broadcast_message(ce_web_interface_t *web_interface);

/**
 * Create LTM stats message
 * @param web_interface Web interface instance
 * @return LTM stats message
 */
ce_web_message_t *ce_web_create_ltm_stats_message(ce_web_interface_t *web_interface);

/**
 * Create neural stats message
 * @param web_interface Web interface instance
 * @return Neural stats message
 */
ce_web_message_t *ce_web_create_neural_stats_message(ce_web_interface_t *web_interface);

/**
 * Create CUDA stats message
 * @param web_interface Web interface instance
 * @return CUDA stats message
 */
ce_web_message_t *ce_web_create_cuda_stats_message(ce_web_interface_t *web_interface);

/**
 * Create reasoning trace message
 * @param web_interface Web interface instance
 * @return Reasoning trace message
 */
ce_web_message_t *ce_web_create_reasoning_trace_message(ce_web_interface_t *web_interface);

/* ============================================================================
 * Client Management Functions
 * ============================================================================ */

/**
 * Add client to pool
 * @param client_pool Client pool
 * @param socket_fd Client socket
 * @param ip_address Client IP address
 * @return Client ID or NULL on error
 */
char *ce_web_add_client(ce_web_client_pool_t *client_pool, int socket_fd,
                        const char *ip_address);

/**
 * Remove client from pool
 * @param client_pool Client pool
 * @param client_id Client ID
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_remove_client(ce_web_client_pool_t *client_pool,
                                const char *client_id);

/**
 * Get client by ID
 * @param client_pool Client pool
 * @param client_id Client ID
 * @return Client or NULL if not found
 */
ce_web_client_t *ce_web_get_client(ce_web_client_pool_t *client_pool,
                                   const char *client_id);

/**
 * Update client subscriptions
 * @param client Client to update
 * @param subscriptions Subscription preferences
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_update_client_subscriptions(ce_web_client_t *client,
                                              const struct {
                                                  bool system_status;
                                                  bool working_memory;
                                                  bool broadcasts;
                                                  bool ltm_stats;
                                                  bool reasoning_trace;
                                                  bool neural_stats;
                                                  bool cuda_stats;
                                              } *subscriptions);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Generate unique client ID
 * @return Client ID (caller must free)
 */
char *ce_web_generate_client_id(void);

/**
 * Validate JSON message
 * @param json_input JSON input
 * @return True if valid JSON
 */
bool ce_web_validate_json(const char *json_input);

/**
 * Extract command from JSON message
 * @param json_input JSON input
 * @param command Output command
 * @param max_length Maximum command length
 * @return CE_SUCCESS on success, error code otherwise
 */
ce_error_t ce_web_extract_command(const char *json_input, char *command,
                                  size_t max_length);

/**
 * Create error message
 * @param error_code Error code
 * @param error_message Error message
 * @return Error message
 */
ce_web_message_t *ce_web_create_error_message(int error_code,
                                              const char *error_message);

/**
 * Create heartbeat message
 * @return Heartbeat message
 */
ce_web_message_t *ce_web_create_heartbeat_message(void);

#endif /* CE_WEB_INTERFACE_H */
