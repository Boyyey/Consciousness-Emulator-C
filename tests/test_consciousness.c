/**
 * Consciousness Emulator - Test Suite
 * 
 * Comprehensive test suite for all CE modules including unit tests,
 * integration tests, and performance benchmarks.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "../include/consciousness.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

/* ============================================================================
 * Test Configuration
 * ============================================================================ */

#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("FAIL: %s\n", message); \
            return 1; \
        } else { \
            printf("PASS: %s\n", message); \
        } \
    } while(0)

#define TEST_ASSERT_EQ(expected, actual, message) \
    TEST_ASSERT((expected) == (actual), message)

#define TEST_ASSERT_NEQ(expected, actual, message) \
    TEST_ASSERT((expected) != (actual), message)

#define TEST_ASSERT_NULL(ptr, message) \
    TEST_ASSERT((ptr) == NULL, message)

#define TEST_ASSERT_NOT_NULL(ptr, message) \
    TEST_ASSERT((ptr) != NULL, message)

/* ============================================================================
 * Test Utilities
 * ============================================================================ */

static int test_count = 0;
static int pass_count = 0;
static int fail_count = 0;

static void test_init(void) {
    test_count = 0;
    pass_count = 0;
    fail_count = 0;
    printf("=== Consciousness Emulator Test Suite ===\n");
    printf("Author: AmirHosseinRasti\n");
    printf("Version: %s\n\n", ce_get_version());
}

static void test_summary(void) {
    printf("\n=== Test Summary ===\n");
    printf("Total tests: %d\n", test_count);
    printf("Passed: %d\n", pass_count);
    printf("Failed: %d\n", fail_count);
    printf("Success rate: %.1f%%\n", test_count > 0 ? (100.0 * pass_count / test_count) : 0.0);
}

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ============================================================================
 * Core System Tests
 * ============================================================================ */

static int test_core_system(void) {
    printf("--- Core System Tests ---\n");
    
    /* Test initialization */
    ce_error_t result = ce_init(10.0);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "System initialization");
    
    /* Test double initialization */
    result = ce_init(10.0);
    TEST_ASSERT_EQ(CE_ERROR_INVALID_STATE, result, "Double initialization should fail");
    
    /* Test shutdown */
    result = ce_shutdown();
    TEST_ASSERT_EQ(CE_SUCCESS, result, "System shutdown");
    
    /* Test operations after shutdown */
    result = ce_tick();
    TEST_ASSERT_EQ(CE_ERROR_INVALID_STATE, result, "Operations after shutdown should fail");
    
    return 0;
}

/* ============================================================================
 * Item Management Tests
 * ============================================================================ */

static int test_item_management(void) {
    printf("--- Item Management Tests ---\n");
    
    /* Test item creation */
    ce_item_t *item = ce_item_create(CE_ITEM_TYPE_BELIEF, "Test belief", 0.8f);
    TEST_ASSERT_NOT_NULL(item, "Item creation");
    TEST_ASSERT_EQ(CE_ITEM_TYPE_BELIEF, item->type, "Item type");
    TEST_ASSERT_EQ(0.8f, item->confidence, "Item confidence");
    TEST_ASSERT_NOT_NULL(item->content, "Item content");
    TEST_ASSERT_EQ(0, strcmp("Test belief", item->content), "Item content string");
    
    /* Test item cloning */
    ce_item_t *clone = ce_item_clone(item);
    TEST_ASSERT_NOT_NULL(clone, "Item cloning");
    TEST_ASSERT_NEQ(item->id, clone->id, "Cloned item should have different ID");
    TEST_ASSERT_EQ(item->type, clone->type, "Cloned item type");
    TEST_ASSERT_EQ(0, strcmp(item->content, clone->content), "Cloned item content");
    
    /* Test item updates */
    ce_item_update_saliency(item, 0.9f);
    TEST_ASSERT_EQ(0.9f, item->saliency, "Item saliency update");
    
    ce_item_update_confidence(item, 0.95f);
    TEST_ASSERT_EQ(0.95f, item->confidence, "Item confidence update");
    
    /* Test item with embedding */
    float embedding[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    ce_item_t *item_with_embedding = ce_item_create_with_embedding(
        CE_ITEM_TYPE_QUESTION, "Test question", embedding, 4, 0.7f);
    TEST_ASSERT_NOT_NULL(item_with_embedding, "Item with embedding creation");
    TEST_ASSERT_NOT_NULL(item_with_embedding->embedding, "Item embedding pointer");
    TEST_ASSERT_EQ(4, item_with_embedding->embedding_dim, "Item embedding dimension");
    TEST_ASSERT_EQ(0.1f, item_with_embedding->embedding[0], "Item embedding value 0");
    TEST_ASSERT_EQ(0.4f, item_with_embedding->embedding[3], "Item embedding value 3");
    
    /* Cleanup */
    ce_item_free(item);
    ce_item_free(clone);
    ce_item_free(item_with_embedding);
    
    return 0;
}

/* ============================================================================
 * Item List Tests
 * ============================================================================ */

static int test_item_lists(void) {
    printf("--- Item List Tests ---\n");
    
    /* Test list creation */
    ce_item_list_t *list = ce_item_list_create(10);
    TEST_ASSERT_NOT_NULL(list, "Item list creation");
    TEST_ASSERT_EQ(0, list->count, "Empty list count");
    TEST_ASSERT_EQ(10, list->capacity, "List capacity");
    
    /* Test adding items */
    ce_item_t *item1 = ce_item_create(CE_ITEM_TYPE_BELIEF, "Item 1", 0.5f);
    ce_item_t *item2 = ce_item_create(CE_ITEM_TYPE_QUESTION, "Item 2", 0.7f);
    ce_item_t *item3 = ce_item_create(CE_ITEM_TYPE_ANSWER, "Item 3", 0.9f);
    
    ce_item_update_saliency(item1, 0.3f);
    ce_item_update_saliency(item2, 0.8f);
    ce_item_update_saliency(item3, 0.6f);
    
    ce_error_t result = ce_item_list_add(list, item1);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "Add item 1");
    TEST_ASSERT_EQ(1, list->count, "List count after adding item 1");
    
    result = ce_item_list_add(list, item2);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "Add item 2");
    TEST_ASSERT_EQ(2, list->count, "List count after adding item 2");
    
    result = ce_item_list_add(list, item3);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "Add item 3");
    TEST_ASSERT_EQ(3, list->count, "List count after adding item 3");
    
    /* Test top-k retrieval */
    ce_item_t *top_items[3];
    size_t top_count = ce_item_list_topk(list, 2, top_items);
    TEST_ASSERT_EQ(2, top_count, "Top-k count");
    TEST_ASSERT_EQ(item2, top_items[0], "Top item should be item 2 (highest saliency)");
    TEST_ASSERT_EQ(item3, top_items[1], "Second item should be item 3");
    
    /* Test item removal */
    result = ce_item_list_remove(list, item2);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "Remove item 2");
    TEST_ASSERT_EQ(2, list->count, "List count after removing item 2");
    
    /* Cleanup */
    ce_item_list_free(list);
    ce_item_free(item1);
    ce_item_free(item2);
    ce_item_free(item3);
    
    return 0;
}

/* ============================================================================
 * Working Memory Tests
 * ============================================================================ */

static int test_working_memory(void) {
    printf("--- Working Memory Tests ---\n");
    
    /* Initialize system for WM tests */
    ce_init(10.0);
    
    /* Test WM creation */
    ce_working_memory_t *wm = ce_wm_create(5);
    TEST_ASSERT_NOT_NULL(wm, "Working memory creation");
    
    /* Test adding items */
    ce_item_t *item1 = ce_item_create(CE_ITEM_TYPE_BELIEF, "Belief 1", 0.8f);
    ce_item_t *item2 = ce_item_create(CE_ITEM_TYPE_QUESTION, "Question 1", 0.6f);
    ce_item_t *item3 = ce_item_create(CE_ITEM_TYPE_ANSWER, "Answer 1", 0.9f);
    
    ce_item_update_saliency(item1, 0.7f);
    ce_item_update_saliency(item2, 0.5f);
    ce_item_update_saliency(item3, 0.8f);
    
    ce_error_t result = ce_wm_add(wm, item1);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "Add item 1 to WM");
    
    result = ce_wm_add(wm, item2);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "Add item 2 to WM");
    
    result = ce_wm_add(wm, item3);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "Add item 3 to WM");
    
    /* Test getting items */
    const ce_item_list_t *items = ce_wm_get_items(wm);
    TEST_ASSERT_NOT_NULL(items, "Get WM items");
    TEST_ASSERT_EQ(3, items->count, "WM item count");
    
    /* Test top-k retrieval */
    ce_item_t *top_items[3];
    size_t top_count = ce_wm_get_topk(wm, 2, top_items);
    TEST_ASSERT_EQ(2, top_count, "WM top-k count");
    TEST_ASSERT_EQ(item3, top_items[0], "Top WM item should be item 3");
    TEST_ASSERT_EQ(item1, top_items[1], "Second WM item should be item 1");
    
    /* Test item access */
    ce_item_t *accessed = ce_wm_access(wm, item2->id);
    TEST_ASSERT_EQ(item2, accessed, "Access item 2");
    
    /* Test WM update (decay) */
    result = ce_wm_update(wm);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "WM update");
    
    /* Test WM statistics */
    struct {
        size_t capacity;
        size_t count;
        size_t active_slots;
        double total_saliency;
        double avg_saliency;
        double max_saliency;
        double min_saliency;
        uint64_t total_items_added;
        uint64_t total_items_removed;
        uint64_t total_accesses;
    } stats;
    
    result = ce_wm_get_stats(wm, &stats);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "Get WM stats");
    TEST_ASSERT_EQ(5, stats.capacity, "WM capacity");
    TEST_ASSERT_EQ(3, stats.count, "WM count");
    TEST_ASSERT_EQ(1, stats.total_accesses, "WM total accesses");
    
    /* Cleanup */
    ce_wm_free(wm);
    ce_item_free(item1);
    ce_item_free(item2);
    ce_item_free(item3);
    ce_shutdown();
    
    return 0;
}

/* ============================================================================
 * Math Utilities Tests
 * ============================================================================ */

static int test_math_utils(void) {
    printf("--- Math Utilities Tests ---\n");
    
    /* Test vector operations */
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float c[4];
    
    /* Test dot product */
    float dot = ce_dot_product(a, b, 4);
    float expected_dot = 1.0f*2.0f + 2.0f*3.0f + 3.0f*4.0f + 4.0f*5.0f;
    TEST_ASSERT_EQ(expected_dot, dot, "Dot product");
    
    /* Test L2 norm */
    float norm = ce_l2_norm(a, 4);
    float expected_norm = sqrtf(1.0f + 4.0f + 9.0f + 16.0f);
    TEST_ASSERT_EQ(expected_norm, norm, "L2 norm");
    
    /* Test cosine similarity */
    float cosine = ce_cosine_similarity(a, b, 4);
    float expected_cosine = dot / (norm * ce_l2_norm(b, 4));
    TEST_ASSERT_EQ(expected_cosine, cosine, "Cosine similarity");
    
    /* Test vector addition */
    ce_vector_add(a, b, c, 4);
    TEST_ASSERT_EQ(3.0f, c[0], "Vector addition element 0");
    TEST_ASSERT_EQ(5.0f, c[1], "Vector addition element 1");
    TEST_ASSERT_EQ(7.0f, c[2], "Vector addition element 2");
    TEST_ASSERT_EQ(9.0f, c[3], "Vector addition element 3");
    
    /* Test vector scaling */
    ce_vector_scale(a, 2.0f, c, 4);
    TEST_ASSERT_EQ(2.0f, c[0], "Vector scaling element 0");
    TEST_ASSERT_EQ(4.0f, c[1], "Vector scaling element 1");
    TEST_ASSERT_EQ(6.0f, c[2], "Vector scaling element 2");
    TEST_ASSERT_EQ(8.0f, c[3], "Vector scaling element 3");
    
    /* Test activation functions */
    float x = 1.0f;
    float sigmoid_result = ce_sigmoid(x);
    float expected_sigmoid = 1.0f / (1.0f + expf(-x));
    TEST_ASSERT_EQ(expected_sigmoid, sigmoid_result, "Sigmoid function");
    
    float relu_result = ce_relu(x);
    TEST_ASSERT_EQ(x, relu_result, "ReLU function (positive)");
    
    float relu_neg_result = ce_relu(-x);
    TEST_ASSERT_EQ(0.0f, relu_neg_result, "ReLU function (negative)");
    
    /* Test random number generation */
    ce_random_init(12345);
    float rand1 = ce_random_float();
    float rand2 = ce_random_float();
    TEST_ASSERT_NEQ(rand1, rand2, "Random number generation");
    TEST_ASSERT(rand1 >= 0.0f && rand1 < 1.0f, "Random number range 1");
    TEST_ASSERT(rand2 >= 0.0f && rand2 < 1.0f, "Random number range 2");
    
    return 0;
}

/* ============================================================================
 * Performance Tests
 * ============================================================================ */

static int test_performance(void) {
    printf("--- Performance Tests ---\n");
    
    /* Test vector operations performance */
    const size_t vector_size = 1000;
    const size_t iterations = 10000;
    
    float *a = malloc(vector_size * sizeof(float));
    float *b = malloc(vector_size * sizeof(float));
    float *c = malloc(vector_size * sizeof(float));
    
    /* Initialize vectors */
    for (size_t i = 0; i < vector_size; i++) {
        a[i] = (float)i / vector_size;
        b[i] = (float)(i + 1) / vector_size;
    }
    
    /* Test dot product performance */
    double start_time = get_time_ms();
    for (size_t i = 0; i < iterations; i++) {
        ce_dot_product(a, b, vector_size);
    }
    double end_time = get_time_ms();
    double dot_time = end_time - start_time;
    
    printf("Dot product: %.2f ms for %zu iterations (%.2f us per operation)\n",
           dot_time, iterations, dot_time * 1000.0 / iterations);
    
    /* Test vector addition performance */
    start_time = get_time_ms();
    for (size_t i = 0; i < iterations; i++) {
        ce_vector_add(a, b, c, vector_size);
    }
    end_time = get_time_ms();
    double add_time = end_time - start_time;
    
    printf("Vector addition: %.2f ms for %zu iterations (%.2f us per operation)\n",
           add_time, iterations, add_time * 1000.0 / iterations);
    
    /* Test working memory performance */
    ce_init(100.0);
    ce_working_memory_t *wm = ce_wm_create(100);
    
    start_time = get_time_ms();
    for (size_t i = 0; i < 1000; i++) {
        char content[64];
        snprintf(content, sizeof(content), "Item %zu", i);
        ce_item_t *item = ce_item_create(CE_ITEM_TYPE_BELIEF, content, 0.5f);
        ce_item_update_saliency(item, (float)i / 1000.0f);
        ce_wm_add(wm, item);
        ce_item_free(item);
    }
    end_time = get_time_ms();
    double wm_time = end_time - start_time;
    
    printf("Working memory: %.2f ms for 1000 items (%.2f us per item)\n",
           wm_time, wm_time * 1000.0 / 1000);
    
    /* Cleanup */
    free(a);
    free(b);
    free(c);
    ce_wm_free(wm);
    ce_shutdown();
    
    return 0;
}

/* ============================================================================
 * Integration Tests
 * ============================================================================ */

static int test_integration(void) {
    printf("--- Integration Tests ---\n");
    
    /* Test full system integration */
    ce_init(20.0);
    
    /* Get global components */
    ce_working_memory_t *wm = ce_kernel_get_global_wm();
    TEST_ASSERT_NOT_NULL(wm, "Global working memory");
    
    ce_workspace_t *workspace = ce_kernel_get_global_workspace();
    TEST_ASSERT_NOT_NULL(workspace, "Global workspace");
    
    ce_long_term_memory_t *ltm = ce_kernel_get_global_ltm();
    TEST_ASSERT_NOT_NULL(ltm, "Global long-term memory");
    
    /* Test adding items to WM */
    ce_item_t *belief = ce_item_create(CE_ITEM_TYPE_BELIEF, "The sky is blue", 0.9f);
    ce_item_t *question = ce_item_create(CE_ITEM_TYPE_QUESTION, "What color is the sky?", 0.8f);
    
    ce_error_t result = ce_wm_add(wm, belief);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "Add belief to global WM");
    
    result = ce_wm_add(wm, question);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "Add question to global WM");
    
    /* Test workspace processing */
    result = ce_workspace_process(workspace);
    TEST_ASSERT_EQ(CE_SUCCESS, result, "Workspace processing");
    
    /* Test LTM storage */
    result = ce_ltm_store_episode(ltm, belief, "Initial belief about sky color");
    TEST_ASSERT_EQ(CE_SUCCESS, result, "Store episode in LTM");
    
    /* Test system tick */
    result = ce_tick();
    TEST_ASSERT_EQ(CE_SUCCESS, result, "System tick");
    
    /* Test multiple ticks */
    for (int i = 0; i < 5; i++) {
        result = ce_tick();
        TEST_ASSERT_EQ(CE_SUCCESS, result, "Multiple system ticks");
    }
    
    /* Cleanup */
    ce_item_free(belief);
    ce_item_free(question);
    ce_shutdown();
    
    return 0;
}

/* ============================================================================
 * Error Handling Tests
 * ============================================================================ */

static int test_error_handling(void) {
    printf("--- Error Handling Tests ---\n");
    
    /* Test null pointer handling */
    ce_item_t *item = ce_item_create(CE_ITEM_TYPE_BELIEF, NULL, 0.5f);
    TEST_ASSERT_NULL(item, "Item creation with NULL content should fail");
    
    ce_working_memory_t *wm = ce_wm_create(0);
    TEST_ASSERT_NULL(wm, "WM creation with 0 capacity should fail");
    
    /* Test invalid parameters */
    ce_error_t result = ce_wm_add(NULL, NULL);
    TEST_ASSERT_EQ(CE_ERROR_NULL_POINTER, result, "WM add with NULL parameters");
    
    /* Test operations on uninitialized system */
    result = ce_tick();
    TEST_ASSERT_EQ(CE_ERROR_INVALID_STATE, result, "Tick on uninitialized system");
    
    /* Test error string function */
    const char *error_str = ce_error_string(CE_ERROR_NULL_POINTER);
    TEST_ASSERT_NOT_NULL(error_str, "Error string function");
    TEST_ASSERT(strlen(error_str) > 0, "Error string not empty");
    
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    
    test_init();
    
    int result = 0;
    
    /* Run all test suites */
    result += test_core_system();
    result += test_item_management();
    result += test_item_lists();
    result += test_working_memory();
    result += test_math_utils();
    result += test_performance();
    result += test_integration();
    result += test_error_handling();
    
    test_summary();
    
    return result;
}
