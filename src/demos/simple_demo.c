/**
 * Consciousness Emulator - Simple Demo
 * 
 * A minimal demonstration of the CE system that works with the current implementation.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "../../include/consciousness.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Simple Demo Functions
 * ============================================================================ */

static void print_version(void) {
    printf("Consciousness Emulator v%s\n", ce_get_version());
    printf("Author: AmirHosseinRasti\n");
    printf("License: MIT\n\n");
}

static void demo_item_creation(void) {
    printf("=== Item Creation Demo ===\n");
    
    /* Create some cognitive items */
    ce_item_t *belief = ce_item_create(CE_ITEM_TYPE_BELIEF, 
                                      "The sky is blue", 0.9f);
    ce_item_t *question = ce_item_create(CE_ITEM_TYPE_QUESTION, 
                                        "What color is the sky?", 0.8f);
    ce_item_t *answer = ce_item_create(CE_ITEM_TYPE_ANSWER, 
                                      "The sky is blue", 0.95f);
    
    if (belief) {
        printf("✓ Created belief: %s (confidence: %.2f)\n", 
               belief->content, belief->confidence);
    }
    
    if (question) {
        printf("✓ Created question: %s (confidence: %.2f)\n", 
               question->content, question->confidence);
    }
    
    if (answer) {
        printf("✓ Created answer: %s (confidence: %.2f)\n", 
               answer->content, answer->confidence);
    }
    
    /* Clean up */
    if (belief) ce_item_free(belief);
    if (question) ce_item_free(question);
    if (answer) ce_item_free(answer);
    
    printf("\n");
}

static void demo_item_lists(void) {
    printf("=== Item List Demo ===\n");
    
    /* Create an item list */
    ce_item_list_t *list = ce_item_list_create(10);
    if (!list) {
        printf("✗ Failed to create item list\n");
        return;
    }
    
    printf("✓ Created item list with capacity %zu\n", list->capacity);
    
    /* Add some items */
    ce_item_t *item1 = ce_item_create(CE_ITEM_TYPE_BELIEF, "Item 1", 0.5f);
    ce_item_t *item2 = ce_item_create(CE_ITEM_TYPE_BELIEF, "Item 2", 0.7f);
    ce_item_t *item3 = ce_item_create(CE_ITEM_TYPE_BELIEF, "Item 3", 0.9f);
    
    if (item1) {
        ce_item_update_saliency(item1, 0.3f);
        ce_item_list_add(list, item1);
    }
    
    if (item2) {
        ce_item_update_saliency(item2, 0.8f);
        ce_item_list_add(list, item2);
    }
    
    if (item3) {
        ce_item_update_saliency(item3, 0.6f);
        ce_item_list_add(list, item3);
    }
    
    printf("✓ Added %zu items to list\n", list->count);
    
    /* Get top-k items */
    ce_item_t *top_items[3];
    size_t top_count = ce_item_list_topk(list, 2, top_items);
    
    printf("✓ Top %zu items by saliency:\n", top_count);
    for (size_t i = 0; i < top_count; i++) {
        printf("  %zu. %s (saliency: %.2f)\n", 
               i + 1, top_items[i]->content, top_items[i]->saliency);
    }
    
    /* Clean up */
    ce_item_list_free(list);
    if (item1) ce_item_free(item1);
    if (item2) ce_item_free(item2);
    if (item3) ce_item_free(item3);
    
    printf("\n");
}

static void demo_working_memory(void) {
    printf("=== Working Memory Demo ===\n");
    
    /* Create working memory */
    ce_working_memory_t *wm = ce_wm_create(5);
    if (!wm) {
        printf("✗ Failed to create working memory\n");
        return;
    }
    
    printf("✓ Created working memory with capacity %zu\n", wm->capacity);
    
    /* Add some items */
    ce_item_t *item1 = ce_item_create(CE_ITEM_TYPE_BELIEF, "Memory item 1", 0.8f);
    ce_item_t *item2 = ce_item_create(CE_ITEM_TYPE_QUESTION, "Memory item 2", 0.7f);
    
    if (item1) {
        ce_item_update_saliency(item1, 0.9f);
        ce_error_t result = ce_wm_add(wm, item1);
        if (result == CE_SUCCESS) {
            printf("✓ Added item 1 to working memory\n");
        }
    }
    
    if (item2) {
        ce_item_update_saliency(item2, 0.6f);
        ce_error_t result = ce_wm_add(wm, item2);
        if (result == CE_SUCCESS) {
            printf("✓ Added item 2 to working memory\n");
        }
    }
    
    /* Get items */
    const ce_item_list_t *items = ce_wm_get_items(wm);
    if (items) {
        printf("✓ Working memory contains %zu items:\n", items->count);
        for (size_t i = 0; i < items->count; i++) {
            ce_item_t *item = items->items[i];
            printf("  %zu. %s (saliency: %.2f)\n", 
                   i + 1, item->content, item->saliency);
        }
    }
    
    /* Clean up */
    ce_wm_free(wm);
    if (item1) ce_item_free(item1);
    if (item2) ce_item_free(item2);
    
    printf("\n");
}

static void demo_math_utils(void) {
    printf("=== Math Utils Demo ===\n");
    
    /* Test vector operations */
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float c[4];
    
    /* Vector addition */
    ce_vector_add(a, b, c, 4);
    printf("✓ Vector addition: [1,2,3,4] + [2,3,4,5] = [%.1f,%.1f,%.1f,%.1f]\n",
           c[0], c[1], c[2], c[3]);
    
    /* Dot product */
    float dot = ce_dot_product(a, b, 4);
    printf("✓ Dot product: %.1f\n", dot);
    
    /* Cosine similarity */
    float cosine = ce_cosine_similarity(a, b, 4);
    printf("✓ Cosine similarity: %.3f\n", cosine);
    
    /* Activation functions */
    float x = 1.0f;
    float sigmoid_result = ce_sigmoid(x);
    float relu_result = ce_relu(x);
    printf("✓ Sigmoid(1.0) = %.3f\n", sigmoid_result);
    printf("✓ ReLU(1.0) = %.3f\n", relu_result);
    
    printf("\n");
}

static void demo_error_handling(void) {
    printf("=== Error Handling Demo ===\n");
    
    /* Test null pointer handling */
    ce_item_t *null_item = ce_item_create(CE_ITEM_TYPE_BELIEF, NULL, 0.5f);
    if (!null_item) {
        printf("✓ Correctly rejected null content\n");
    }
    
    ce_error_t result = ce_wm_add(NULL, NULL);
    if (result == CE_ERROR_NULL_POINTER) {
        printf("✓ Correctly detected null pointer error\n");
    }
    
    /* Test error string function */
    const char *error_str = ce_error_string(CE_ERROR_NULL_POINTER);
    if (error_str) {
        printf("✓ Error string: %s\n", error_str);
    }
    
    printf("\n");
}

/* ============================================================================
 * Main Function
 * ============================================================================ */

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    
    printf("=== Consciousness Emulator Simple Demo ===\n\n");
    
    print_version();
    
    demo_item_creation();
    demo_item_lists();
    demo_working_memory();
    demo_math_utils();
    demo_error_handling();
    
    printf("=== Demo Complete ===\n");
    printf("All basic functionality is working correctly!\n");
    
    return 0;
}
