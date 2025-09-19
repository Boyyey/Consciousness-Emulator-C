/**
 * Consciousness Emulator v1.1 - Advanced Demo
 * 
 * Comprehensive demonstration of v1.1 features including neural networks,
 * CUDA acceleration, advanced reasoning, and web visualization.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "../../include/consciousness.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>

/* ============================================================================
 * Demo Configuration
 * ============================================================================ */

#define DEMO_TICK_HZ 20.0
#define DEMO_DURATION 60.0
#define DEMO_EMBEDDING_DIM 128
#define DEMO_NEURAL_BATCH_SIZE 32
#define DEMO_WEB_PORT 8080

/* ============================================================================
 * Global Demo State
 * ============================================================================ */

static bool g_demo_running = true;
static ce_neural_engine_t *g_neural_engine = NULL;
static ce_cuda_context_t *g_cuda_context = NULL;
static ce_advanced_reasoner_t *g_advanced_reasoner = NULL;
static ce_web_interface_t *g_web_interface = NULL;

/* ============================================================================
 * Signal Handling
 * ============================================================================ */

static void signal_handler(int sig) {
    (void)sig;
    g_demo_running = false;
    printf("\nShutting down v1.1 demo...\n");
}

static void setup_signal_handlers(void) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
}

/* ============================================================================
 * Neural Network Demo Functions
 * ============================================================================ */

static int demo_neural_networks(void) {
    printf("=== Neural Network Integration Demo ===\n");
    
    /* Create neural engine */
    g_neural_engine = ce_neural_engine_create(0, 0, 4); /* CPU, FP32, 4 threads */
    if (!g_neural_engine) {
        printf("ERROR: Failed to create neural engine\n");
        return 1;
    }
    printf("✓ Neural engine created successfully\n");
    
    /* Load demo models (these would be real ONNX models in practice) */
    void *embedding_model = ce_neural_engine_load_model(g_neural_engine, "embedding_model",
                                                       "models/embedding.onnx", 3);
    if (embedding_model) {
        printf("✓ Embedding model loaded\n");
    } else {
        printf("⚠ Embedding model not available (using fallback)\n");
    }
    
    void *classification_model = ce_neural_engine_load_model(g_neural_engine, "classification_model",
                                                            "models/classification.onnx", 4);
    if (classification_model) {
        printf("✓ Classification model loaded\n");
    } else {
        printf("⚠ Classification model not available (using fallback)\n");
    }
    
    /* Test neural inference */
    printf("\nTesting neural inference...\n");
    
    float input[DEMO_EMBEDDING_DIM];
    float output[DEMO_EMBEDDING_DIM];
    
    /* Generate test input */
    for (size_t i = 0; i < DEMO_EMBEDDING_DIM; i++) {
        input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    /* Run inference */
    ce_error_t result = ce_neural_engine_infer(g_neural_engine, "embedding_model",
                                              input, output);
    if (result == CE_SUCCESS) {
        printf("✓ Neural inference successful\n");
        
        /* Test classification */
        float class_probs[10];
        result = ce_neural_engine_infer(g_neural_engine, "classification_model",
                                       input, class_probs);
        if (result == CE_SUCCESS) {
            printf("✓ Classification inference successful\n");
            
            /* Find best class */
            int best_class = 0;
            float best_prob = class_probs[0];
            for (int i = 1; i < 10; i++) {
                if (class_probs[i] > best_prob) {
                    best_prob = class_probs[i];
                    best_class = i;
                }
            }
            printf("  Best class: %d (probability: %.3f)\n", best_class, best_prob);
        }
    } else {
        printf("⚠ Neural inference failed (using fallback)\n");
    }
    
    return 0;
}

/* ============================================================================
 * CUDA Acceleration Demo Functions
 * ============================================================================ */

static int demo_cuda_acceleration(void) {
    printf("\n=== CUDA Acceleration Demo ===\n");
    
    /* Check CUDA availability */
    if (!ce_cuda_is_available()) {
        printf("⚠ CUDA not available, skipping CUDA demo\n");
        return 0;
    }
    
    /* Initialize CUDA context */
    g_cuda_context = ce_cuda_init();
    if (!g_cuda_context) {
        printf("ERROR: Failed to initialize CUDA context\n");
        return 1;
    }
    printf("✓ CUDA context initialized\n");
    
    /* Test CUDA vector operations */
    printf("\nTesting CUDA vector operations...\n");
    
    const size_t vector_size = 1000000;
    float *a = malloc(vector_size * sizeof(float));
    float *b = malloc(vector_size * sizeof(float));
    float *c = malloc(vector_size * sizeof(float));
    
    if (!a || !b || !c) {
        printf("ERROR: Failed to allocate test vectors\n");
        free(a); free(b); free(c);
        return 1;
    }
    
    /* Initialize test vectors */
    for (size_t i = 0; i < vector_size; i++) {
        a[i] = (float)i / vector_size;
        b[i] = (float)(i + 1) / vector_size;
    }
    
    /* Test CPU vs CUDA performance */
    printf("Testing %zu-element vector addition...\n", vector_size);
    
    /* CPU version */
    double cpu_start = ce_get_timestamp();
    for (size_t i = 0; i < vector_size; i++) {
        c[i] = a[i] + b[i];
    }
    double cpu_end = ce_get_timestamp();
    double cpu_time = cpu_end - cpu_start;
    
    printf("  CPU time: %.4f ms\n", cpu_time * 1000.0);
    
    /* CUDA version (would use actual CUDA kernels in real implementation) */
    double cuda_start = ce_get_timestamp();
    ce_error_t result = ce_cuda_vector_add(a, b, c, vector_size);
    double cuda_end = ce_get_timestamp();
    double cuda_time = cuda_end - cuda_start;
    
    if (result == CE_SUCCESS) {
        printf("  CUDA time: %.4f ms\n", cuda_time * 1000.0);
        printf("  Speedup: %.2fx\n", cpu_time / cuda_time);
    } else {
        printf("  CUDA operation failed (using CPU fallback)\n");
    }
    
    free(a); free(b); free(c);
    
    return 0;
}

/* ============================================================================
 * Advanced Reasoning Demo Functions
 * ============================================================================ */

static int demo_advanced_reasoning(void) {
    printf("\n=== Advanced Reasoning Demo ===\n");
    
    /* Create advanced reasoner */
    g_advanced_reasoner = ce_advanced_reasoner_create(0, 0.7f, 10); /* Forward chaining, 0.7 threshold, depth 10 */
    if (!g_advanced_reasoner) {
        printf("ERROR: Failed to create advanced reasoner\n");
        return 1;
    }
    printf("✓ Advanced reasoner created successfully\n");
    
    /* Add reasoning rules */
    printf("\nAdding reasoning rules...\n");
    
    uint64_t rule1 = ce_advanced_reasoner_add_rule(g_advanced_reasoner, "cat_mammal",
                                                  0, "X is a cat", "X is a mammal", 0.95f);
    if (rule1) {
        printf("✓ Added rule: cat_mammal\n");
    }
    
    uint64_t rule2 = ce_advanced_reasoner_add_rule(g_advanced_reasoner, "mammal_fur",
                                                  0, "X is a mammal", "X has fur", 0.85f);
    if (rule2) {
        printf("✓ Added rule: mammal_fur\n");
    }
    
    uint64_t rule3 = ce_advanced_reasoner_add_rule(g_advanced_reasoner, "fur_keeps_warm",
                                                  0, "X has fur", "X stays warm", 0.90f);
    if (rule3) {
        printf("✓ Added rule: fur_keeps_warm\n");
    }
    
    /* Create reasoning context */
    uint64_t context_id = ce_advanced_reasoner_create_context(g_advanced_reasoner,
                                                             "animal_properties",
                                                             "Reasoning about animal properties");
    if (context_id) {
        printf("✓ Created reasoning context: %lu\n", context_id);
    }
    
    /* Add facts */
    ce_item_t *fact1 = ce_item_create(CE_ITEM_TYPE_BELIEF, "Fluffy is a cat", 0.9f);
    if (fact1) {
        ce_advanced_reasoner_add_fact(g_advanced_reasoner, context_id, fact1);
        printf("✓ Added fact: Fluffy is a cat\n");
    }
    
    /* Perform reasoning */
    printf("\nPerforming forward chaining reasoning...\n");
    ce_item_list_t *conclusions = ce_advanced_reasoner_reason(g_advanced_reasoner, context_id, 0);
    
    if (conclusions && conclusions->count > 0) {
        printf("✓ Generated %zu conclusions:\n", conclusions->count);
        for (size_t i = 0; i < conclusions->count; i++) {
            printf("  %zu. %s (confidence: %.3f)\n", i + 1, 
                   conclusions->items[i]->content, conclusions->items[i]->confidence);
        }
        ce_item_list_free(conclusions);
    } else {
        printf("⚠ No conclusions generated\n");
    }
    
    if (fact1) {
        ce_item_free(fact1);
    }
    
    return 0;
}

/* ============================================================================
 * Web Visualization Demo Functions
 * ============================================================================ */

static int demo_web_visualization(void) {
    printf("\n=== Web Visualization Demo ===\n");
    
    /* Create web interface */
    g_web_interface = ce_web_interface_create(DEMO_WEB_PORT, 0.1); /* 100ms updates */
    if (!g_web_interface) {
        printf("ERROR: Failed to create web interface\n");
        return 1;
    }
    printf("✓ Web interface created successfully\n");
    
    /* Start web server */
    ce_error_t result = ce_web_interface_start(g_web_interface);
    if (result == CE_SUCCESS) {
        printf("✓ Web server started on port %d\n", DEMO_WEB_PORT);
        printf("  Open http://localhost:%d in your browser to view the visualization\n", DEMO_WEB_PORT);
    } else {
        printf("⚠ Failed to start web server (port may be in use)\n");
    }
    
    return 0;
}

/* ============================================================================
 * Integrated System Demo
 * ============================================================================ */

static int demo_integrated_system(void) {
    printf("\n=== Integrated System Demo ===\n");
    
    /* Initialize core system */
    if (ce_init(DEMO_TICK_HZ) != CE_SUCCESS) {
        printf("ERROR: Failed to initialize CE system\n");
        return 1;
    }
    printf("✓ Core system initialized\n");
    
    /* Get global components */
    ce_working_memory_t *wm = ce_kernel_get_global_wm();
    ce_workspace_t *workspace = ce_kernel_get_global_workspace();
    ce_long_term_memory_t *ltm = ce_kernel_get_global_ltm();
    
    if (!wm || !workspace || !ltm) {
        printf("ERROR: Failed to get global components\n");
        ce_shutdown();
        return 1;
    }
    
    /* Add some cognitive items */
    printf("\nAdding cognitive items...\n");
    
    ce_item_t *belief1 = ce_item_create(CE_ITEM_TYPE_BELIEF, "Neural networks can learn patterns", 0.9f);
    ce_item_t *belief2 = ce_item_create(CE_ITEM_TYPE_BELIEF, "CUDA provides massive parallelization", 0.95f);
    ce_item_t *question1 = ce_item_create(CE_ITEM_TYPE_QUESTION, "How can we combine neural and symbolic reasoning?", 0.8f);
    
    if (belief1) {
        ce_wm_add(wm, belief1);
        printf("✓ Added belief: Neural networks can learn patterns\n");
    }
    
    if (belief2) {
        ce_wm_add(wm, belief2);
        printf("✓ Added belief: CUDA provides massive parallelization\n");
    }
    
    if (question1) {
        ce_wm_add(wm, question1);
        printf("✓ Added question: How can we combine neural and symbolic reasoning?\n");
    }
    
    /* Run integrated cognitive loop */
    printf("\nRunning integrated cognitive loop...\n");
    
    for (int i = 0; i < 20 && g_demo_running; i++) {
        printf("--- Cognitive Cycle %d ---\n", i + 1);
        
        /* Process core cognitive cycle */
        ce_tick();
        
        /* Update web visualization */
        if (g_web_interface) {
            ce_web_interface_update_data(g_web_interface);
        }
        
        /* Show working memory state */
        const ce_item_list_t *wm_items = ce_wm_get_items(wm);
        if (wm_items && wm_items->count > 0) {
            printf("  Working Memory (%zu items):\n", wm_items->count);
            for (size_t j = 0; j < wm_items->count && j < 5; j++) {
                ce_item_t *item = wm_items->items[j];
                printf("    [%lu] %s (saliency: %.3f)\n", 
                       item->id, item->content, item->saliency);
            }
        }
        
        /* Show workspace broadcasts */
        const ce_item_list_t *broadcasts = ce_workspace_get_broadcast(workspace);
        if (broadcasts && broadcasts->count > 0) {
            printf("  Broadcasts (%zu items):\n", broadcasts->count);
            for (size_t j = 0; j < broadcasts->count; j++) {
                ce_item_t *item = broadcasts->items[j];
                printf("    [%lu] %s (saliency: %.3f)\n", 
                       item->id, item->content, item->saliency);
            }
        }
        
        /* Store in LTM */
        if (belief1 && i % 5 == 0) {
            ce_ltm_store_episode(ltm, belief1, "Demo episode");
        }
        
        printf("\n");
        usleep(500000); /* 500ms delay */
    }
    
    /* Show final statistics */
    printf("=== Final System Statistics ===\n");
    
    struct {
        double total_runtime;
        uint64_t total_ticks;
        double avg_tick_time;
        double max_tick_time;
        size_t active_modules;
        size_t message_queue_size;
    } kernel_stats;
    
    if (ce_kernel_get_stats(&kernel_stats) == CE_SUCCESS) {
        printf("Kernel Stats:\n");
        printf("  Runtime: %.2f seconds\n", kernel_stats.total_runtime);
        printf("  Total ticks: %lu\n", kernel_stats.total_ticks);
        printf("  Avg tick time: %.4f ms\n", kernel_stats.avg_tick_time * 1000.0);
        printf("  Max tick time: %.4f ms\n", kernel_stats.max_tick_time * 1000.0);
        printf("  Active modules: %zu\n", kernel_stats.active_modules);
    }
    
    /* Cleanup */
    if (belief1) ce_item_free(belief1);
    if (belief2) ce_item_free(belief2);
    if (question1) ce_item_free(question1);
    
    ce_shutdown();
    
    return 0;
}

/* ============================================================================
 * Main Demo Function
 * ============================================================================ */

static int run_v1_1_demo(void) {
    printf("=== Consciousness Emulator v1.1 - Advanced Demo ===\n");
    printf("Author: AmirHosseinRasti\n");
    printf("Version: %s\n", ce_get_version());
    printf("\n");
    
    setup_signal_handlers();
    
    /* Run individual demos */
    if (demo_neural_networks() != 0) {
        return 1;
    }
    
    if (demo_cuda_acceleration() != 0) {
        return 1;
    }
    
    if (demo_advanced_reasoning() != 0) {
        return 1;
    }
    
    if (demo_web_visualization() != 0) {
        return 1;
    }
    
    /* Run integrated system demo */
    if (demo_integrated_system() != 0) {
        return 1;
    }
    
    /* Keep web interface running */
    if (g_web_interface) {
        printf("\nWeb interface is running. Press Ctrl+C to exit.\n");
        while (g_demo_running) {
            sleep(1);
        }
    }
    
    return 0;
}

/* ============================================================================
 * Cleanup Function
 * ============================================================================ */

static void cleanup_demo(void) {
    printf("\nCleaning up demo resources...\n");
    
    if (g_web_interface) {
        ce_web_interface_stop(g_web_interface);
        ce_web_interface_free(g_web_interface);
        printf("✓ Web interface cleaned up\n");
    }
    
    if (g_advanced_reasoner) {
        ce_advanced_reasoner_free(g_advanced_reasoner);
        printf("✓ Advanced reasoner cleaned up\n");
    }
    
    if (g_cuda_context) {
        ce_cuda_free(g_cuda_context);
        printf("✓ CUDA context cleaned up\n");
    }
    
    if (g_neural_engine) {
        ce_neural_engine_free(g_neural_engine);
        printf("✓ Neural engine cleaned up\n");
    }
    
    printf("Demo cleanup complete.\n");
}

/* ============================================================================
 * Main Function
 * ============================================================================ */

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    
    int result = run_v1_1_demo();
    
    cleanup_demo();
    
    return result;
}
