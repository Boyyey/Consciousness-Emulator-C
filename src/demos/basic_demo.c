/**
 * Consciousness Emulator - Basic Demo
 * 
 * A simple demonstration of the CE system showing basic cognitive
 * operations including working memory, attention, and reasoning.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "../../include/consciousness.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ============================================================================
 * Demo Configuration
 * ============================================================================ */

#define DEMO_TICK_HZ 10.0
#define DEMO_DURATION 30.0
#define DEMO_EMBEDDING_DIM 64

/* ============================================================================
 * Demo Data
 * ============================================================================ */

static const char *demo_facts[] = {
    "The sky is blue during the day",
    "Water boils at 100 degrees Celsius",
    "Cats are mammals",
    "The Earth orbits around the Sun",
    "Gravity pulls objects downward",
    "Light travels faster than sound",
    "Plants produce oxygen through photosynthesis",
    "The human brain contains billions of neurons",
    "Mathematics is the language of science",
    "Time is a dimension of spacetime"
};

static const char *demo_questions[] = {
    "What color is the sky?",
    "At what temperature does water boil?",
    "Are cats mammals?",
    "What does the Earth orbit around?",
    "What force pulls objects downward?",
    "What travels faster, light or sound?",
    "How do plants produce oxygen?",
    "What does the human brain contain?",
    "What is the language of science?",
    "What is time?"
};

/* ============================================================================
 * Demo Functions
 * ============================================================================ */

/**
 * Generate a random embedding for demo purposes
 */
static void generate_demo_embedding(float *embedding, size_t dim, const char *text) {
    if (!embedding || !text) {
        return;
    }
    
    /* Simple hash-based embedding generation */
    uint32_t hash = 0;
    for (const char *p = text; *p; p++) {
        hash = hash * 31 + *p;
    }
    
    ce_generate_random_embedding(embedding, dim, hash);
}

/**
 * Create a demo item with random embedding
 */
static ce_item_t *create_demo_item(ce_item_type_t type, const char *content, float confidence) {
    float *embedding = malloc(DEMO_EMBEDDING_DIM * sizeof(float));
    if (!embedding) {
        return NULL;
    }
    
    generate_demo_embedding(embedding, DEMO_EMBEDDING_DIM, content);
    
    ce_item_t *item = ce_item_create_with_embedding(type, content, embedding, 
                                                   DEMO_EMBEDDING_DIM, confidence);
    
    free(embedding);
    return item;
}

/**
 * Print working memory contents
 */
static void print_working_memory(const ce_working_memory_t *wm) {
    if (!wm) {
        return;
    }
    
    const ce_item_list_t *items = ce_wm_get_items(wm);
    if (!items || items->count == 0) {
        printf("  Working Memory: (empty)\n");
        return;
    }
    
    printf("  Working Memory (%zu items):\n", items->count);
    for (size_t i = 0; i < items->count; i++) {
        ce_item_t *item = items->items[i];
        printf("    [%llu] %s (saliency: %.3f, confidence: %.3f)\n",
               (unsigned long long)item->id, item->content, item->saliency, item->confidence);
    }
}

/**
 * Print system statistics
 */
static void print_system_stats(void) {
    struct {
        double total_runtime;
        uint64_t total_ticks;
        double avg_tick_time;
        double max_tick_time;
        size_t active_modules;
        size_t message_queue_size;
    } stats;
    
    if (ce_kernel_get_stats(&stats) == CE_SUCCESS) {
        printf("  System Stats:\n");
        printf("    Runtime: %.2f seconds\n", stats.total_runtime);
        printf("    Total ticks: %llu\n", (unsigned long long)stats.total_ticks);
        printf("    Avg tick time: %.4f ms\n", stats.avg_tick_time * 1000.0);
        printf("    Max tick time: %.4f ms\n", stats.max_tick_time * 1000.0);
        printf("    Active modules: %zu\n", stats.active_modules);
        printf("    Message queue size: %zu\n", stats.message_queue_size);
    }
}

/**
 * Run the basic demo
 */
static int run_basic_demo(void) {
    printf("=== Consciousness Emulator - Basic Demo ===\n");
    printf("Author: AmirHosseinRasti\n");
    printf("Version: %s\n", ce_get_version());
    printf("\n");
    
    /* Initialize the system */
    printf("Initializing Consciousness Emulator...\n");
    if (ce_init(DEMO_TICK_HZ) != CE_SUCCESS) {
        printf("ERROR: Failed to initialize CE system\n");
        return 1;
    }
    printf("✓ System initialized successfully\n\n");
    
    /* Get global working memory */
    ce_working_memory_t *wm = ce_kernel_get_global_wm();
    if (!wm) {
        printf("ERROR: Failed to get global working memory\n");
        ce_shutdown();
        return 1;
    }
    
    /* Add some demo facts to working memory */
    printf("Adding demo facts to working memory...\n");
    for (size_t i = 0; i < 5; i++) {
        ce_item_t *fact = create_demo_item(CE_ITEM_TYPE_BELIEF, demo_facts[i], 0.9f);
        if (fact) {
            ce_wm_add(wm, fact);
            printf("  Added: %s\n", demo_facts[i]);
            ce_item_free(fact);
        }
    }
    printf("\n");
    
    /* Run the system for a few ticks */
    printf("Running cognitive loop for 5 ticks...\n");
    for (int i = 0; i < 5; i++) {
        printf("--- Tick %d ---\n", i + 1);
        
        /* Process a tick */
        if (ce_tick() != CE_SUCCESS) {
            printf("ERROR: Failed to process tick\n");
            break;
        }
        
        /* Print working memory state */
        print_working_memory(wm);
        
        /* Add a question every other tick */
        if (i % 2 == 1 && (size_t)(i / 2) < sizeof(demo_questions) / sizeof(demo_questions[0])) {
            ce_item_t *question = create_demo_item(CE_ITEM_TYPE_QUESTION, 
                                                  demo_questions[i / 2], 0.8f);
            if (question) {
                ce_wm_add(wm, question);
                printf("  Added question: %s\n", demo_questions[i / 2]);
                ce_item_free(question);
            }
        }
        
        printf("\n");
        usleep(100000); /* 100ms delay */
    }
    
    /* Print final statistics */
    printf("=== Final System State ===\n");
    print_working_memory(wm);
    print_system_stats();
    
    /* Shutdown the system */
    printf("\nShutting down Consciousness Emulator...\n");
    if (ce_shutdown() != CE_SUCCESS) {
        printf("ERROR: Failed to shutdown CE system\n");
        return 1;
    }
    printf("✓ System shutdown successfully\n");
    
    return 0;
}

/**
 * Run the interactive demo
 */
static int run_interactive_demo(void) {
    printf("=== Consciousness Emulator - Interactive Demo ===\n");
    printf("Type 'quit' to exit, 'stats' for statistics, or any text to add to working memory\n\n");
    
    /* Initialize the system */
    if (ce_init(DEMO_TICK_HZ) != CE_SUCCESS) {
        printf("ERROR: Failed to initialize CE system\n");
        return 1;
    }
    
    ce_working_memory_t *wm = ce_kernel_get_global_wm();
    if (!wm) {
        printf("ERROR: Failed to get global working memory\n");
        ce_shutdown();
        return 1;
    }
    
    char input[1024];
    int tick_count = 0;
    
    while (1) {
        printf("CE> ");
        fflush(stdout);
        
        if (!fgets(input, sizeof(input), stdin)) {
            break;
        }
        
        /* Remove newline */
        input[strcspn(input, "\n")] = '\0';
        
        if (strcmp(input, "quit") == 0) {
            break;
        } else if (strcmp(input, "stats") == 0) {
            print_working_memory(wm);
            print_system_stats();
        } else if (strlen(input) > 0) {
            /* Add input as a belief */
            ce_item_t *item = create_demo_item(CE_ITEM_TYPE_BELIEF, input, 0.7f);
            if (item) {
                ce_wm_add(wm, item);
                printf("Added to working memory: %s\n", input);
                ce_item_free(item);
            }
            
            /* Process a few ticks */
            for (int i = 0; i < 3; i++) {
                ce_tick();
                tick_count++;
            }
            
            print_working_memory(wm);
        }
        
        printf("\n");
    }
    
    printf("Total ticks processed: %d\n", tick_count);
    
    /* Shutdown */
    ce_shutdown();
    return 0;
}

/* ============================================================================
 * Main Function
 * ============================================================================ */

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--interactive") == 0) {
        return run_interactive_demo();
    } else {
        return run_basic_demo();
    }
}
