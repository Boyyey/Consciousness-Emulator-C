/**
 * Consciousness Emulator - Message Queue Implementation
 * 
 * Thread-safe message queue for inter-module communication.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* ============================================================================
 * Message Queue Implementation
 * ============================================================================ */

ce_message_queue_t *ce_message_queue_create(size_t capacity) {
    ce_message_queue_t *queue = malloc(sizeof(ce_message_queue_t));
    if (!queue) {
        return NULL;
    }
    
    queue->messages = calloc(capacity, sizeof(ce_message_t));
    if (!queue->messages) {
        free(queue);
        return NULL;
    }
    
    queue->count = 0;
    queue->capacity = capacity;
    queue->head = 0;
    queue->tail = 0;
    
    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        free(queue->messages);
        free(queue);
        return NULL;
    }
    
    if (pthread_cond_init(&queue->condition, NULL) != 0) {
        pthread_mutex_destroy(&queue->mutex);
        free(queue->messages);
        free(queue);
        return NULL;
    }
    
    return queue;
}

void ce_message_queue_destroy(ce_message_queue_t *queue) {
    if (!queue) {
        return;
    }
    
    /* Free any remaining messages */
    for (size_t i = 0; i < queue->count; i++) {
        size_t index = (queue->head + i) % queue->capacity;
        if (queue->messages[index].payload) {
            ce_item_free(queue->messages[index].payload);
        }
    }
    
    pthread_cond_destroy(&queue->condition);
    pthread_mutex_destroy(&queue->mutex);
    free(queue->messages);
    free(queue);
}

ce_error_t ce_message_queue_enqueue(ce_message_queue_t *queue, const ce_message_t *message) {
    if (!queue || !message) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->count >= queue->capacity) {
        pthread_mutex_unlock(&queue->mutex);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Copy message to queue */
    queue->messages[queue->tail] = *message;
    
    /* Clone payload if present */
    if (message->payload) {
        queue->messages[queue->tail].payload = ce_item_clone(message->payload);
        if (!queue->messages[queue->tail].payload) {
            pthread_mutex_unlock(&queue->mutex);
            return CE_ERROR_OUT_OF_MEMORY;
        }
    }
    
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->count++;
    
    pthread_cond_signal(&queue->condition);
    pthread_mutex_unlock(&queue->mutex);
    
    return CE_SUCCESS;
}

ce_error_t ce_message_queue_dequeue(ce_message_queue_t *queue, ce_message_t *message, double timeout_seconds) {
    if (!queue || !message) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&queue->mutex);
    
    /* Wait for message if queue is empty */
    if (queue->count == 0) {
        if (timeout_seconds > 0) {
            struct timespec timeout;
            timeout.tv_sec = (time_t)timeout_seconds;
            timeout.tv_nsec = (long)((timeout_seconds - timeout.tv_sec) * 1000000000);
            
            int result = pthread_cond_timedwait(&queue->condition, &queue->mutex, &timeout);
            if (result == ETIMEDOUT) {
                pthread_mutex_unlock(&queue->mutex);
                return CE_ERROR_UNKNOWN; /* Timeout */
            }
        } else if (timeout_seconds == 0) {
            /* Non-blocking */
            pthread_mutex_unlock(&queue->mutex);
            return CE_ERROR_UNKNOWN; /* No message available */
        } else {
            /* Blocking wait */
            pthread_cond_wait(&queue->condition, &queue->mutex);
        }
    }
    
    if (queue->count == 0) {
        pthread_mutex_unlock(&queue->mutex);
        return CE_ERROR_UNKNOWN;
    }
    
    /* Copy message from queue */
    *message = queue->messages[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->count--;
    
    pthread_mutex_unlock(&queue->mutex);
    
    return CE_SUCCESS;
}

size_t ce_message_queue_size(const ce_message_queue_t *queue) {
    if (!queue) {
        return 0;
    }
    
    pthread_mutex_lock((pthread_mutex_t *)&queue->mutex);
    size_t size = queue->count;
    pthread_mutex_unlock((pthread_mutex_t *)&queue->mutex);
    
    return size;
}

/* ============================================================================
 * Message Creation and Management
 * ============================================================================ */

ce_message_t *ce_message_create(ce_message_type_t type, const char *source, 
                               const char *target, ce_item_t *payload) {
    ce_message_t *message = malloc(sizeof(ce_message_t));
    if (!message) {
        return NULL;
    }
    
    static uint64_t message_id_counter = 1;
    
    message->id = message_id_counter++;
    message->type = type;
    message->timestamp = ce_get_timestamp();
    
    if (source) {
        strncpy(message->source_module, source, sizeof(message->source_module) - 1);
        message->source_module[sizeof(message->source_module) - 1] = '\0';
    } else {
        message->source_module[0] = '\0';
    }
    
    if (target) {
        strncpy(message->target_module, target, sizeof(message->target_module) - 1);
        message->target_module[sizeof(message->target_module) - 1] = '\0';
    } else {
        message->target_module[0] = '\0';
    }
    
    message->payload = payload;
    message->user_data = NULL;
    message->user_data_size = 0;
    
    return message;
}

void ce_message_free(ce_message_t *message) {
    if (!message) {
        return;
    }
    
    if (message->payload) {
        ce_item_free(message->payload);
    }
    
    if (message->user_data) {
        free(message->user_data);
    }
    
    free(message);
}
