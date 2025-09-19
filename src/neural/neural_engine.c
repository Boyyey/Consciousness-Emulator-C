/**
 * Consciousness Emulator v1.1 - Neural Network Engine Implementation
 * 
 * ONNX Runtime integration with fallback implementations for systems
 * without ONNX Runtime available.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "neural_engine.h"
#include "../utils/math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <dlfcn.h>

/* ============================================================================
 * ONNX Runtime Integration
 * ============================================================================ */

/* ONNX Runtime function pointers */
typedef void* (*OrtCreateSessionOptionsFunc)(void);
typedef void* (*OrtCreateSessionFunc)(void*, const char*, void*, void*);
typedef int (*OrtRunFunc)(void*, const char*, const void**, size_t, const char**, size_t, void**);
typedef void (*OrtReleaseSessionFunc)(void*);
typedef void (*OrtReleaseSessionOptionsFunc)(void*);

static struct {
    void *onnx_library;
    OrtCreateSessionOptionsFunc CreateSessionOptions;
    OrtCreateSessionFunc CreateSession;
    OrtRunFunc Run;
    OrtReleaseSessionFunc ReleaseSession;
    OrtReleaseSessionOptionsFunc ReleaseSessionOptions;
    bool is_available;
} g_onnx_runtime = {0};

/* ============================================================================
 * Fallback Neural Network Implementation
 * ============================================================================ */

typedef struct {
    float *weights;
    float *biases;
    size_t input_size;
    size_t output_size;
    size_t hidden_size;
    int num_layers;
} ce_fallback_model_t;

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * Initialize ONNX Runtime
 */
static bool init_onnx_runtime(void) {
    if (g_onnx_runtime.is_available) {
        return true;
    }
    
    /* Try to load ONNX Runtime library */
    g_onnx_runtime.onnx_library = dlopen("libonnxruntime.so", RTLD_LAZY);
    if (!g_onnx_runtime.onnx_library) {
        g_onnx_runtime.onnx_library = dlopen("onnxruntime.dll", RTLD_LAZY);
    }
    
    if (!g_onnx_runtime.onnx_library) {
        return false; /* ONNX Runtime not available */
    }
    
    /* Load function pointers */
    g_onnx_runtime.CreateSessionOptions = (OrtCreateSessionOptionsFunc)
        dlsym(g_onnx_runtime.onnx_library, "OrtCreateSessionOptions");
    g_onnx_runtime.CreateSession = (OrtCreateSessionFunc)
        dlsym(g_onnx_runtime.onnx_library, "OrtCreateSession");
    g_onnx_runtime.Run = (OrtRunFunc)
        dlsym(g_onnx_runtime.onnx_library, "OrtRun");
    g_onnx_runtime.ReleaseSession = (OrtReleaseSessionFunc)
        dlsym(g_onnx_runtime.onnx_library, "OrtReleaseSession");
    g_onnx_runtime.ReleaseSessionOptions = (OrtReleaseSessionOptionsFunc)
        dlsym(g_onnx_runtime.onnx_library, "OrtReleaseSessionOptions");
    
    if (!g_onnx_runtime.CreateSessionOptions || !g_onnx_runtime.CreateSession ||
        !g_onnx_runtime.Run || !g_onnx_runtime.ReleaseSession ||
        !g_onnx_runtime.ReleaseSessionOptions) {
        dlclose(g_onnx_runtime.onnx_library);
        return false;
    }
    
    g_onnx_runtime.is_available = true;
    return true;
}

/**
 * Create fallback neural network model
 */
static ce_fallback_model_t *create_fallback_model(size_t input_size, size_t output_size) {
    ce_fallback_model_t *model = malloc(sizeof(ce_fallback_model_t));
    if (!model) {
        return NULL;
    }
    
    model->input_size = input_size;
    model->output_size = output_size;
    model->hidden_size = (input_size + output_size) / 2;
    model->num_layers = 2;
    
    /* Allocate weights and biases */
    size_t total_weights = input_size * model->hidden_size + 
                          model->hidden_size * output_size;
    model->weights = calloc(total_weights, sizeof(float));
    model->biases = calloc(model->hidden_size + output_size, sizeof(float));
    
    if (!model->weights || !model->biases) {
        free(model->weights);
        free(model->biases);
        free(model);
        return NULL;
    }
    
    /* Initialize with random weights */
    ce_random_init((uint32_t)time(NULL));
    for (size_t i = 0; i < total_weights; i++) {
        model->weights[i] = ce_random_float_range(-0.1f, 0.1f);
    }
    
    return model;
}

/**
 * Run fallback neural network inference
 */
static ce_error_t run_fallback_inference(ce_fallback_model_t *model,
                                        const float *input,
                                        float *output) {
    if (!model || !input || !output) {
        return CE_ERROR_NULL_POINTER;
    }
    
    /* Allocate hidden layer */
    float *hidden = malloc(model->hidden_size * sizeof(float));
    if (!hidden) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Forward pass through hidden layer */
    for (size_t i = 0; i < model->hidden_size; i++) {
        hidden[i] = model->biases[i];
        for (size_t j = 0; j < model->input_size; j++) {
            hidden[i] += input[j] * model->weights[i * model->input_size + j];
        }
        hidden[i] = ce_relu(hidden[i]); /* ReLU activation */
    }
    
    /* Forward pass through output layer */
    for (size_t i = 0; i < model->output_size; i++) {
        output[i] = model->biases[model->hidden_size + i];
        for (size_t j = 0; j < model->hidden_size; j++) {
            output[i] += hidden[j] * model->weights[model->input_size * model->hidden_size + 
                                                   i * model->hidden_size + j];
        }
        output[i] = ce_sigmoid(output[i]); /* Sigmoid activation */
    }
    
    free(hidden);
    return CE_SUCCESS;
}

/**
 * Free fallback model
 */
static void free_fallback_model(ce_fallback_model_t *model) {
    if (model) {
        free(model->weights);
        free(model->biases);
        free(model);
    }
}

/* ============================================================================
 * Public API Implementation
 * ============================================================================ */

ce_neural_engine_t *ce_neural_engine_create(ce_neural_device_t device,
                                           ce_neural_precision_t precision,
                                           int num_threads) {
    ce_neural_engine_t *engine = calloc(1, sizeof(ce_neural_engine_t));
    if (!engine) {
        return NULL;
    }
    
    /* Initialize configuration */
    engine->default_device = device;
    engine->default_precision = precision;
    engine->default_batch_size = CE_NEURAL_DEFAULT_BATCH_SIZE;
    engine->num_threads = num_threads > 0 ? num_threads : CE_NEURAL_DEFAULT_THREADS;
    
    /* Initialize statistics */
    engine->total_inferences = 0;
    engine->total_engine_time = 0.0;
    engine->avg_engine_time = 0.0;
    
    /* Initialize callbacks */
    engine->on_model_loaded = NULL;
    engine->on_inference_complete = NULL;
    engine->callback_user_data = NULL;
    
    /* Initialize synchronization */
    if (pthread_mutex_init(&engine->mutex, NULL) != 0) {
        free(engine);
        return NULL;
    }
    
    /* Initialize ONNX Runtime if available */
    init_onnx_runtime();
    
    return engine;
}

void ce_neural_engine_free(ce_neural_engine_t *engine) {
    if (!engine) {
        return;
    }
    
    /* Unload all models */
    for (size_t i = 0; i < engine->model_count; i++) {
        ce_neural_model_t *model = &engine->models[i];
        if (model->is_loaded) {
            if (g_onnx_runtime.is_available && model->session) {
                g_onnx_runtime.ReleaseSession(model->session);
            }
            if (model->fallback_model) {
                free_fallback_model(model->fallback_model);
            }
        }
        if (model->model_path) {
            free(model->model_path);
        }
    }
    
    /* Destroy synchronization */
    pthread_mutex_destroy(&engine->mutex);
    
    /* Close ONNX Runtime library */
    if (g_onnx_runtime.onnx_library) {
        dlclose(g_onnx_runtime.onnx_library);
    }
    
    free(engine);
}

ce_neural_model_t *ce_neural_engine_load_model(ce_neural_engine_t *engine,
                                               const char *name,
                                               const char *model_path,
                                               ce_neural_model_type_t type) {
    if (!engine || !name || !model_path) {
        return NULL;
    }
    
    pthread_mutex_lock(&engine->mutex);
    
    if (engine->model_count >= CE_NEURAL_MAX_MODELS) {
        pthread_mutex_unlock(&engine->mutex);
        return NULL;
    }
    
    ce_neural_model_t *model = &engine->models[engine->model_count];
    
    /* Initialize model structure */
    strncpy(model->name, name, sizeof(model->name) - 1);
    model->name[sizeof(model->name) - 1] = '\0';
    
    model->model_path = malloc(strlen(model_path) + 1);
    if (!model->model_path) {
        pthread_mutex_unlock(&engine->mutex);
        return NULL;
    }
    strcpy(model->model_path, model_path);
    
    model->type = type;
    model->device = engine->default_device;
    model->precision = engine->default_precision;
    model->batch_size = engine->default_batch_size;
    
    /* Initialize statistics */
    model->inference_count = 0;
    model->total_inference_time = 0.0;
    model->avg_inference_time = 0.0;
    model->max_inference_time = 0.0;
    
    /* Try to load ONNX model first */
    if (g_onnx_runtime.is_available) {
        model->session_options = g_onnx_runtime.CreateSessionOptions();
        if (model->session_options) {
            model->session = g_onnx_runtime.CreateSession(NULL, model_path, 
                                                         model->session_options, NULL);
            if (model->session) {
                model->is_loaded = true;
                model->input_size = 512;  /* Default sizes - would be extracted from model */
                model->output_size = 256;
            }
        }
    }
    
    /* Fallback to custom implementation if ONNX fails */
    if (!model->is_loaded) {
        model->fallback_model = create_fallback_model(512, 256);
        if (model->fallback_model) {
            model->is_loaded = true;
            model->input_size = model->fallback_model->input_size;
            model->output_size = model->fallback_model->output_size;
        }
    }
    
    if (model->is_loaded) {
        engine->model_count++;
        
        /* Call model loaded callback */
        if (engine->on_model_loaded) {
            engine->on_model_loaded(model, engine->callback_user_data);
        }
    }
    
    pthread_mutex_unlock(&engine->mutex);
    
    return model->is_loaded ? model : NULL;
}

ce_error_t ce_neural_engine_unload_model(ce_neural_engine_t *engine,
                                         const char *model_name) {
    if (!engine || !model_name) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&engine->mutex);
    
    for (size_t i = 0; i < engine->model_count; i++) {
        ce_neural_model_t *model = &engine->models[i];
        
        if (strcmp(model->name, model_name) == 0) {
            /* Unload model */
            if (model->is_loaded) {
                if (g_onnx_runtime.is_available && model->session) {
                    g_onnx_runtime.ReleaseSession(model->session);
                }
                if (model->fallback_model) {
                    free_fallback_model(model->fallback_model);
                }
            }
            
            if (model->model_path) {
                free(model->model_path);
            }
            
            /* Shift remaining models */
            for (size_t j = i; j < engine->model_count - 1; j++) {
                engine->models[j] = engine->models[j + 1];
            }
            
            engine->model_count--;
            pthread_mutex_unlock(&engine->mutex);
            return CE_SUCCESS;
        }
    }
    
    pthread_mutex_unlock(&engine->mutex);
    return CE_ERROR_UNKNOWN; /* Model not found */
}

ce_error_t ce_neural_engine_infer(ce_neural_engine_t *engine,
                                  const char *model_name,
                                  const float *input,
                                  float *output) {
    if (!engine || !model_name || !input || !output) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&engine->mutex);
    
    ce_neural_model_t *model = NULL;
    for (size_t i = 0; i < engine->model_count; i++) {
        if (strcmp(engine->models[i].name, model_name) == 0) {
            model = &engine->models[i];
            break;
        }
    }
    
    if (!model || !model->is_loaded) {
        pthread_mutex_unlock(&engine->mutex);
        return CE_ERROR_UNKNOWN; /* Model not found */
    }
    
    double inference_start = ce_get_timestamp();
    ce_error_t result = CE_SUCCESS;
    
    /* Run inference */
    if (g_onnx_runtime.is_available && model->session) {
        /* ONNX Runtime inference */
        const void *input_tensors[] = {input};
        const char *input_names[] = {"input"};
        const char *output_names[] = {"output"};
        void *output_tensors[] = {output};
        
        int ort_result = g_onnx_runtime.Run(model->session, NULL,
                                          input_tensors, 1,
                                          input_names, 1,
                                          output_names, 1,
                                          output_tensors);
        if (ort_result != 0) {
            result = CE_ERROR_UNKNOWN;
        }
    } else if (model->fallback_model) {
        /* Fallback inference */
        result = run_fallback_inference(model->fallback_model, input, output);
    } else {
        result = CE_ERROR_UNKNOWN;
    }
    
    double inference_end = ce_get_timestamp();
    double inference_time = inference_end - inference_start;
    
    /* Update statistics */
    if (result == CE_SUCCESS) {
        model->inference_count++;
        model->total_inference_time += inference_time;
        model->avg_inference_time = model->total_inference_time / model->inference_count;
        
        if (inference_time > model->max_inference_time) {
            model->max_inference_time = inference_time;
        }
        
        engine->total_inferences++;
        engine->total_engine_time += inference_time;
        engine->avg_engine_time = engine->total_engine_time / engine->total_inferences;
        
        /* Call inference complete callback */
        if (engine->on_inference_complete) {
            engine->on_inference_complete(model, input, output, inference_time,
                                        engine->callback_user_data);
        }
    }
    
    pthread_mutex_unlock(&engine->mutex);
    
    return result;
}

ce_error_t ce_neural_engine_infer_batch(ce_neural_engine_t *engine,
                                        const char *model_name,
                                        const float **inputs,
                                        float **outputs,
                                        size_t batch_size) {
    if (!engine || !model_name || !inputs || !outputs || batch_size == 0) {
        return CE_ERROR_NULL_POINTER;
    }
    
    ce_error_t result = CE_SUCCESS;
    
    /* Process batch sequentially */
    for (size_t i = 0; i < batch_size; i++) {
        ce_error_t batch_result = ce_neural_engine_infer(engine, model_name,
                                                        inputs[i], outputs[i]);
        if (batch_result != CE_SUCCESS) {
            result = batch_result;
            break;
        }
    }
    
    return result;
}

ce_neural_model_t *ce_neural_engine_get_model(ce_neural_engine_t *engine,
                                              const char *model_name) {
    if (!engine || !model_name) {
        return NULL;
    }
    
    pthread_mutex_lock(&engine->mutex);
    
    for (size_t i = 0; i < engine->model_count; i++) {
        if (strcmp(engine->models[i].name, model_name) == 0) {
            pthread_mutex_unlock(&engine->mutex);
            return &engine->models[i];
        }
    }
    
    pthread_mutex_unlock(&engine->mutex);
    return NULL;
}

ce_error_t ce_neural_engine_get_stats(const ce_neural_engine_t *engine, struct {
    size_t loaded_models;
    uint64_t total_inferences;
    double avg_inference_time;
    double max_inference_time;
    double total_engine_time;
} *stats) {
    if (!engine || !stats) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock((pthread_mutex_t *)&engine->mutex);
    
    stats->loaded_models = engine->model_count;
    stats->total_inferences = engine->total_inferences;
    stats->avg_inference_time = engine->avg_engine_time;
    stats->total_engine_time = engine->total_engine_time;
    
    /* Find max inference time across all models */
    stats->max_inference_time = 0.0;
    for (size_t i = 0; i < engine->model_count; i++) {
        if (engine->models[i].max_inference_time > stats->max_inference_time) {
            stats->max_inference_time = engine->models[i].max_inference_time;
        }
    }
    
    pthread_mutex_unlock((pthread_mutex_t *)&engine->mutex);
    
    return CE_SUCCESS;
}

ce_error_t ce_neural_engine_set_callbacks(ce_neural_engine_t *engine,
                                          void (*on_model_loaded)(const ce_neural_model_t *model, void *user_data),
                                          void (*on_inference_complete)(const ce_neural_model_t *model, 
                                                                       const float *input, const float *output, 
                                                                       double inference_time, void *user_data),
                                          void *user_data) {
    if (!engine) {
        return CE_ERROR_NULL_POINTER;
    }
    
    pthread_mutex_lock(&engine->mutex);
    
    engine->on_model_loaded = on_model_loaded;
    engine->on_inference_complete = on_inference_complete;
    engine->callback_user_data = user_data;
    
    pthread_mutex_unlock(&engine->mutex);
    
    return CE_SUCCESS;
}

/* ============================================================================
 * Specialized Neural Functions Implementation
 * ============================================================================ */

ce_error_t ce_neural_generate_embedding(ce_neural_engine_t *engine,
                                        const ce_item_t *item,
                                        float *embedding,
                                        size_t embedding_dim) {
    if (!engine || !item || !embedding || embedding_dim == 0) {
        return CE_ERROR_NULL_POINTER;
    }
    
    /* Convert item to tensor */
    float *input_tensor = malloc(1024 * sizeof(float));
    if (!input_tensor) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    size_t input_size = ce_neural_item_to_tensor(item, input_tensor, 1024);
    
    /* Run embedding model */
    ce_error_t result = ce_neural_engine_infer(engine, "embedding_model",
                                              input_tensor, embedding);
    
    free(input_tensor);
    return result;
}

ce_error_t ce_neural_classify_item(ce_neural_engine_t *engine,
                                   const ce_item_t *item,
                                   float *probabilities,
                                   size_t num_classes) {
    if (!engine || !item || !probabilities || num_classes == 0) {
        return CE_ERROR_NULL_POINTER;
    }
    
    /* Convert item to tensor */
    float *input_tensor = malloc(1024 * sizeof(float));
    if (!input_tensor) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    size_t input_size = ce_neural_item_to_tensor(item, input_tensor, 1024);
    
    /* Run classification model */
    ce_error_t result = ce_neural_engine_infer(engine, "classification_model",
                                              input_tensor, probabilities);
    
    /* Apply softmax to get probabilities */
    if (result == CE_SUCCESS) {
        ce_softmax(probabilities, probabilities, num_classes);
    }
    
    free(input_tensor);
    return result;
}

ce_error_t ce_neural_predict_state(ce_neural_engine_t *engine,
                                   const float *current_state,
                                   float *predicted_state,
                                   size_t state_dim) {
    if (!engine || !current_state || !predicted_state || state_dim == 0) {
        return CE_ERROR_NULL_POINTER;
    }
    
    return ce_neural_engine_infer(engine, "prediction_model",
                                 current_state, predicted_state);
}

ce_error_t ce_neural_compute_association(ce_neural_engine_t *engine,
                                         const ce_item_t *item1,
                                         const ce_item_t *item2,
                                         float *association_strength) {
    if (!engine || !item1 || !item2 || !association_strength) {
        return CE_ERROR_NULL_POINTER;
    }
    
    /* Create combined input tensor */
    float *input_tensor = malloc(2048 * sizeof(float));
    if (!input_tensor) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    size_t size1 = ce_neural_item_to_tensor(item1, input_tensor, 1024);
    size_t size2 = ce_neural_item_to_tensor(item2, &input_tensor[1024], 1024);
    
    /* Run association model */
    ce_error_t result = ce_neural_engine_infer(engine, "association_model",
                                              input_tensor, association_strength);
    
    free(input_tensor);
    return result;
}

ce_error_t ce_neural_recognize_pattern(ce_neural_engine_t *engine,
                                       const ce_item_t **sequence,
                                       float *pattern_confidence) {
    if (!engine || !sequence || !pattern_confidence) {
        return CE_ERROR_NULL_POINTER;
    }
    
    /* Convert sequence to tensor */
    float *input_tensor = malloc(4096 * sizeof(float));
    if (!input_tensor) {
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    size_t offset = 0;
    for (size_t i = 0; i < sequence_length && offset < 4096; i++) {
        size_t item_size = ce_neural_item_to_tensor(sequence[i], 
                                                   &input_tensor[offset], 
                                                   4096 - offset);
        offset += item_size;
    }
    
    /* Run pattern recognition model */
    ce_error_t result = ce_neural_engine_infer(engine, "pattern_model",
                                              input_tensor, pattern_confidence);
    
    free(input_tensor);
    return result;
}

/* ============================================================================
 * Utility Functions Implementation
 * ============================================================================ */

ce_error_t ce_neural_get_device_capabilities(ce_neural_device_t device, struct {
    bool available;
    size_t max_memory;
    int compute_capability;
    char device_name[256];
} *capabilities) {
    if (!capabilities) {
        return CE_ERROR_NULL_POINTER;
    }
    
    memset(capabilities, 0, sizeof(*capabilities));
    
    switch (device) {
        case CE_NEURAL_DEVICE_CPU:
            capabilities->available = true;
            capabilities->max_memory = 8ULL * 1024 * 1024 * 1024; /* 8GB */
            strcpy(capabilities->device_name, "CPU");
            break;
            
        case CE_NEURAL_DEVICE_CUDA:
            capabilities->available = false; /* Would check for CUDA availability */
            capabilities->max_memory = 16ULL * 1024 * 1024 * 1024; /* 16GB */
            capabilities->compute_capability = 75; /* RTX 20xx series */
            strcpy(capabilities->device_name, "NVIDIA GPU");
            break;
            
        default:
            capabilities->available = false;
            break;
    }
    
    return CE_SUCCESS;
}

size_t ce_neural_item_to_tensor(const ce_item_t *item, float *tensor, size_t max_size) {
    if (!item || !tensor || max_size == 0) {
        return 0;
    }
    
    size_t offset = 0;
    
    /* Add type as one-hot encoding */
    if (offset + 10 < max_size) {
        for (int i = 0; i < 10; i++) {
            tensor[offset + i] = (i == item->type) ? 1.0f : 0.0f;
        }
        offset += 10;
    }
    
    /* Add confidence */
    if (offset < max_size) {
        tensor[offset] = item->confidence;
        offset++;
    }
    
    /* Add saliency */
    if (offset < max_size) {
        tensor[offset] = item->saliency;
        offset++;
    }
    
    /* Add embedding if available */
    if (item->embedding && item->embedding_dim > 0) {
        size_t embed_size = (item->embedding_dim < max_size - offset) ? 
                           item->embedding_dim : max_size - offset;
        memcpy(&tensor[offset], item->embedding, embed_size * sizeof(float));
        offset += embed_size;
    }
    
    /* Add content hash if no embedding */
    if (!item->embedding && item->content && offset + 4 < max_size) {
        uint32_t hash = 0;
        for (const char *p = item->content; *p; p++) {
            hash = hash * 31 + *p;
        }
        memcpy(&tensor[offset], &hash, sizeof(uint32_t));
        offset += 4;
    }
    
    return offset;
}

ce_item_t *ce_neural_tensor_to_item(const float *tensor, size_t tensor_size,
                                    ce_item_type_t item_type, float confidence) {
    if (!tensor || tensor_size == 0) {
        return NULL;
    }
    
    /* Create item with basic properties */
    ce_item_t *item = ce_item_create(item_type, "Neural generated item", confidence);
    if (!item) {
        return NULL;
    }
    
    /* Set saliency from tensor if available */
    if (tensor_size > 1) {
        item->saliency = tensor[1];
    }
    
    /* Create embedding from tensor */
    if (tensor_size > 2) {
        size_t embed_size = tensor_size - 2;
        item->embedding = malloc(embed_size * sizeof(float));
        if (item->embedding) {
            memcpy(item->embedding, &tensor[2], embed_size * sizeof(float));
            item->embedding_dim = embed_size;
        }
    }
    
    return item;
}

ce_error_t ce_neural_benchmark_model(ce_neural_model_t *model,
                                     size_t num_iterations,
                                     struct {
                                         double avg_inference_time;
                                         double min_inference_time;
                                         double max_inference_time;
                                         double throughput;
                                     } *stats) {
    if (!model || !stats || num_iterations == 0) {
        return CE_ERROR_NULL_POINTER;
    }
    
    /* Create test input and output */
    float *input = malloc(model->input_size * sizeof(float));
    float *output = malloc(model->output_size * sizeof(float));
    
    if (!input || !output) {
        free(input);
        free(output);
        return CE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Initialize test input */
    for (size_t i = 0; i < model->input_size; i++) {
        input[i] = ce_random_float_range(-1.0f, 1.0f);
    }
    
    /* Run benchmark */
    double total_time = 0.0;
    double min_time = 1e9;
    double max_time = 0.0;
    
    for (size_t i = 0; i < num_iterations; i++) {
        double start = ce_get_timestamp();
        
        if (model->fallback_model) {
            run_fallback_inference(model->fallback_model, input, output);
        }
        
        double end = ce_get_timestamp();
        double iter_time = end - start;
        
        total_time += iter_time;
        if (iter_time < min_time) min_time = iter_time;
        if (iter_time > max_time) max_time = iter_time;
    }
    
    /* Calculate statistics */
    stats->avg_inference_time = total_time / num_iterations;
    stats->min_inference_time = min_time;
    stats->max_inference_time = max_time;
    stats->throughput = num_iterations / total_time;
    
    free(input);
    free(output);
    
    return CE_SUCCESS;
}
