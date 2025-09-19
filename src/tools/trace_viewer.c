/**
 * Consciousness Emulator - Trace Viewer
 * 
 * A command-line tool for viewing and analyzing CE system traces.
 * Provides real-time monitoring and historical analysis capabilities.
 * 
 * Author: AmirHosseinRasti
 * License: MIT
 */

#include "../../include/consciousness.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <signal.h>
#include <time.h>

/* ============================================================================
 * Trace Viewer Configuration
 * ============================================================================ */

#define TRACE_VIEWER_VERSION "1.0.0"
#define MAX_TRACE_LINES 10000
#define REFRESH_INTERVAL 1.0

typedef enum {
    VIEW_MODE_REALTIME = 0,
    VIEW_MODE_HISTORICAL,
    VIEW_MODE_STATISTICS,
    VIEW_MODE_EXPORT
} view_mode_t;

typedef struct {
    view_mode_t mode;
    char *trace_file;
    char *output_file;
    bool follow_mode;
    bool verbose;
    double refresh_rate;
    size_t max_lines;
} trace_config_t;

/* ============================================================================
 * Global State
 * ============================================================================ */

static bool g_running = true;
static trace_config_t g_config = {0};

/* ============================================================================
 * Signal Handling
 * ============================================================================ */

static void signal_handler(int sig) {
    (void)sig;
    g_running = false;
    printf("\nShutting down trace viewer...\n");
}

static void setup_signal_handlers(void) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
}

/* ============================================================================
 * Trace Analysis Functions
 * ============================================================================ */

typedef struct {
    uint64_t total_ticks;
    uint64_t total_broadcasts;
    uint64_t total_items;
    double total_runtime;
    double avg_tick_time;
    double max_tick_time;
    size_t peak_wm_size;
    size_t current_wm_size;
} trace_stats_t;

static void analyze_trace_line(const char *line, trace_stats_t *stats) {
    if (!line || !stats) {
        return;
    }
    
    /* Simple trace line parsing - in a real implementation,
     * this would parse JSON or structured log format */
    if (strstr(line, "tick")) {
        stats->total_ticks++;
    } else if (strstr(line, "broadcast")) {
        stats->total_broadcasts++;
    } else if (strstr(line, "item_added")) {
        stats->total_items++;
    } else if (strstr(line, "wm_size")) {
        /* Extract working memory size */
        char *size_str = strstr(line, "wm_size:");
        if (size_str) {
            size_t wm_size = atoi(size_str + 8);
            if (wm_size > stats->peak_wm_size) {
                stats->peak_wm_size = wm_size;
            }
            stats->current_wm_size = wm_size;
        }
    }
}

static void print_trace_stats(const trace_stats_t *stats) {
    printf("\n=== Trace Statistics ===\n");
    printf("Total ticks: %lu\n", stats->total_ticks);
    printf("Total broadcasts: %lu\n", stats->total_broadcasts);
    printf("Total items: %lu\n", stats->total_items);
    printf("Total runtime: %.2f seconds\n", stats->total_runtime);
    printf("Average tick time: %.4f ms\n", stats->avg_tick_time * 1000.0);
    printf("Max tick time: %.4f ms\n", stats->max_tick_time * 1000.0);
    printf("Peak WM size: %zu items\n", stats->peak_wm_size);
    printf("Current WM size: %zu items\n", stats->current_wm_size);
    printf("========================\n\n");
}

/* ============================================================================
 * Real-time Monitoring
 * ============================================================================ */

static void monitor_realtime(void) {
    printf("=== Real-time CE System Monitor ===\n");
    printf("Press Ctrl+C to exit\n\n");
    
    /* Initialize CE system for monitoring */
    if (ce_init(10.0) != CE_SUCCESS) {
        printf("ERROR: Failed to initialize CE system\n");
        return;
    }
    
    ce_working_memory_t *wm = ce_kernel_get_global_wm();
    ce_workspace_t *workspace = ce_kernel_get_global_workspace();
    ce_long_term_memory_t *ltm = ce_kernel_get_global_ltm();
    
    if (!wm || !workspace || !ltm) {
        printf("ERROR: Failed to get global components\n");
        ce_shutdown();
        return;
    }
    
    trace_stats_t stats = {0};
    double last_refresh = ce_get_timestamp();
    
    while (g_running) {
        double current_time = ce_get_timestamp();
        
        if (current_time - last_refresh >= g_config.refresh_rate) {
            /* Clear screen */
            printf("\033[2J\033[H");
            
            /* Print header */
            printf("Consciousness Emulator - Real-time Monitor\n");
            printf("Time: %.2f | Refresh: %.1f Hz\n\n", current_time, 1.0 / g_config.refresh_rate);
            
            /* Get working memory state */
            const ce_item_list_t *wm_items = ce_wm_get_items(wm);
            printf("Working Memory (%zu items):\n", wm_items ? wm_items->count : 0);
            
            if (wm_items && wm_items->count > 0) {
                for (size_t i = 0; i < wm_items->count && i < 10; i++) {
                    ce_item_t *item = wm_items->items[i];
                    printf("  [%lu] %s (saliency: %.3f)\n", 
                           item->id, item->content, item->saliency);
                }
                if (wm_items->count > 10) {
                    printf("  ... and %zu more items\n", wm_items->count - 10);
                }
            }
            printf("\n");
            
            /* Get workspace state */
            const ce_item_list_t *broadcast = ce_workspace_get_broadcast(workspace);
            printf("Current Broadcast (%zu items):\n", broadcast ? broadcast->count : 0);
            
            if (broadcast && broadcast->count > 0) {
                for (size_t i = 0; i < broadcast->count; i++) {
                    ce_item_t *item = broadcast->items[i];
                    printf("  [%lu] %s (saliency: %.3f)\n", 
                           item->id, item->content, item->saliency);
                }
            }
            printf("\n");
            
            /* Get system statistics */
            struct {
                double total_runtime;
                uint64_t total_ticks;
                double avg_tick_time;
                double max_tick_time;
                size_t active_modules;
                size_t message_queue_size;
            } kernel_stats;
            
            if (ce_kernel_get_stats(&kernel_stats) == CE_SUCCESS) {
                printf("System Statistics:\n");
                printf("  Runtime: %.2f seconds\n", kernel_stats.total_runtime);
                printf("  Total ticks: %lu\n", kernel_stats.total_ticks);
                printf("  Avg tick time: %.4f ms\n", kernel_stats.avg_tick_time * 1000.0);
                printf("  Max tick time: %.4f ms\n", kernel_stats.max_tick_time * 1000.0);
                printf("  Active modules: %zu\n", kernel_stats.active_modules);
                printf("  Message queue: %zu\n", kernel_stats.message_queue_size);
            }
            
            /* Get LTM statistics */
            struct {
                size_t total_episodes;
                size_t consolidated_episodes;
                size_t semantic_index_size;
                size_t cluster_count;
                uint64_t total_searches;
                uint64_t total_consolidations;
                double avg_search_time;
                double max_search_time;
            } ltm_stats;
            
            if (ce_ltm_get_stats(ltm, &ltm_stats) == CE_SUCCESS) {
                printf("\nLong-Term Memory:\n");
                printf("  Total episodes: %zu\n", ltm_stats.total_episodes);
                printf("  Consolidated: %zu\n", ltm_stats.consolidated_episodes);
                printf("  Semantic index: %zu\n", ltm_stats.semantic_index_size);
                printf("  Clusters: %zu\n", ltm_stats.cluster_count);
                printf("  Searches: %lu\n", ltm_stats.total_searches);
                printf("  Consolidations: %lu\n", ltm_stats.total_consolidations);
            }
            
            last_refresh = current_time;
        }
        
        /* Process a tick */
        ce_tick();
        
        /* Small delay to prevent excessive CPU usage */
        usleep(10000); /* 10ms */
    }
    
    ce_shutdown();
}

/* ============================================================================
 * Historical Analysis
 * ============================================================================ */

static void analyze_historical(const char *filename) {
    printf("=== Historical Trace Analysis ===\n");
    printf("File: %s\n\n", filename);
    
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("ERROR: Cannot open trace file '%s'\n", filename);
        return;
    }
    
    trace_stats_t stats = {0};
    char line[1024];
    size_t line_count = 0;
    
    printf("Analyzing trace file...\n");
    
    while (fgets(line, sizeof(line), file) && line_count < g_config.max_lines) {
        analyze_trace_line(line, &stats);
        line_count++;
        
        if (line_count % 1000 == 0) {
            printf("Processed %zu lines...\r", line_count);
            fflush(stdout);
        }
    }
    
    printf("\nAnalysis complete. Processed %zu lines.\n", line_count);
    
    print_trace_stats(&stats);
    
    fclose(file);
}

/* ============================================================================
 * Export Functions
 * ============================================================================ */

static void export_trace_data(const char *input_file, const char *output_file) {
    printf("=== Exporting Trace Data ===\n");
    printf("Input: %s\n", input_file);
    printf("Output: %s\n\n", output_file);
    
    FILE *input = fopen(input_file, "r");
    if (!input) {
        printf("ERROR: Cannot open input file '%s'\n", input_file);
        return;
    }
    
    FILE *output = fopen(output_file, "w");
    if (!output) {
        printf("ERROR: Cannot create output file '%s'\n", output_file);
        fclose(input);
        return;
    }
    
    /* Write CSV header */
    fprintf(output, "timestamp,event_type,item_id,item_type,content,saliency,confidence\n");
    
    char line[1024];
    size_t exported_count = 0;
    
    while (fgets(line, sizeof(line), input)) {
        /* Simple export - in a real implementation, this would parse
         * structured trace data and export to CSV/JSON format */
        if (strstr(line, "item_added") || strstr(line, "broadcast")) {
            fprintf(output, "%.6f,%s\n", ce_get_timestamp(), line);
            exported_count++;
        }
    }
    
    printf("Exported %zu events to %s\n", exported_count, output_file);
    
    fclose(input);
    fclose(output);
}

/* ============================================================================
 * Command Line Interface
 * ============================================================================ */

static void print_usage(const char *program_name) {
    printf("Consciousness Emulator Trace Viewer v%s\n", TRACE_VIEWER_VERSION);
    printf("Author: AmirHosseinRasti\n\n");
    printf("Usage: %s [OPTIONS]\n\n", program_name);
    printf("Options:\n");
    printf("  -m, --mode MODE        View mode (realtime|historical|stats|export)\n");
    printf("  -f, --file FILE        Trace file to analyze\n");
    printf("  -o, --output FILE      Output file for export mode\n");
    printf("  -r, --refresh RATE     Refresh rate for realtime mode (Hz)\n");
    printf("  -l, --lines LINES      Maximum lines to process\n");
    printf("  -v, --verbose          Verbose output\n");
    printf("  -h, --help             Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s -m realtime                    # Real-time monitoring\n", program_name);
    printf("  %s -m historical -f trace.log     # Analyze historical trace\n", program_name);
    printf("  %s -m export -f trace.log -o data.csv  # Export to CSV\n", program_name);
}

static int parse_arguments(int argc, char *argv[]) {
    static struct option long_options[] = {
        {"mode", required_argument, 0, 'm'},
        {"file", required_argument, 0, 'f'},
        {"output", required_argument, 0, 'o'},
        {"refresh", required_argument, 0, 'r'},
        {"lines", required_argument, 0, 'l'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    /* Set defaults */
    g_config.mode = VIEW_MODE_REALTIME;
    g_config.refresh_rate = REFRESH_INTERVAL;
    g_config.max_lines = MAX_TRACE_LINES;
    g_config.verbose = false;
    g_config.trace_file = NULL;
    g_config.output_file = NULL;
    
    while ((c = getopt_long(argc, argv, "m:f:o:r:l:vh", long_options, &option_index)) != -1) {
        switch (c) {
            case 'm':
                if (strcmp(optarg, "realtime") == 0) {
                    g_config.mode = VIEW_MODE_REALTIME;
                } else if (strcmp(optarg, "historical") == 0) {
                    g_config.mode = VIEW_MODE_HISTORICAL;
                } else if (strcmp(optarg, "stats") == 0) {
                    g_config.mode = VIEW_MODE_STATISTICS;
                } else if (strcmp(optarg, "export") == 0) {
                    g_config.mode = VIEW_MODE_EXPORT;
                } else {
                    printf("ERROR: Invalid mode '%s'\n", optarg);
                    return 1;
                }
                break;
                
            case 'f':
                g_config.trace_file = strdup(optarg);
                break;
                
            case 'o':
                g_config.output_file = strdup(optarg);
                break;
                
            case 'r':
                g_config.refresh_rate = 1.0 / atof(optarg);
                break;
                
            case 'l':
                g_config.max_lines = atoi(optarg);
                break;
                
            case 'v':
                g_config.verbose = true;
                break;
                
            case 'h':
                print_usage(argv[0]);
                return 0;
                
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    /* Validate configuration */
    if (g_config.mode == VIEW_MODE_HISTORICAL || g_config.mode == VIEW_MODE_EXPORT) {
        if (!g_config.trace_file) {
            printf("ERROR: Trace file required for historical/export mode\n");
            return 1;
        }
    }
    
    if (g_config.mode == VIEW_MODE_EXPORT && !g_config.output_file) {
        printf("ERROR: Output file required for export mode\n");
        return 1;
    }
    
    return 0;
}

/* ============================================================================
 * Main Function
 * ============================================================================ */

int main(int argc, char *argv[]) {
    /* Parse command line arguments */
    if (parse_arguments(argc, argv) != 0) {
        return 1;
    }
    
    /* Setup signal handlers */
    setup_signal_handlers();
    
    /* Run based on mode */
    switch (g_config.mode) {
        case VIEW_MODE_REALTIME:
            monitor_realtime();
            break;
            
        case VIEW_MODE_HISTORICAL:
            analyze_historical(g_config.trace_file);
            break;
            
        case VIEW_MODE_STATISTICS:
            if (g_config.trace_file) {
                analyze_historical(g_config.trace_file);
            } else {
                printf("ERROR: Trace file required for statistics mode\n");
                return 1;
            }
            break;
            
        case VIEW_MODE_EXPORT:
            export_trace_data(g_config.trace_file, g_config.output_file);
            break;
            
        default:
            printf("ERROR: Invalid mode\n");
            return 1;
    }
    
    /* Cleanup */
    if (g_config.trace_file) {
        free(g_config.trace_file);
    }
    if (g_config.output_file) {
        free(g_config.output_file);
    }
    
    return 0;
}
