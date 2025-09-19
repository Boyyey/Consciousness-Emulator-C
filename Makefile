# Consciousness Emulator (CE) - C Implementation
# Author: AmirHosseinRasti
# License: MIT

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O3 -march=native -mtune=native
CFLAGS += -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L
CFLAGS += -ffast-math -funroll-loops -fomit-frame-pointer
CFLAGS += -fopenmp -mavx2 -mfma

# Debug flags
DEBUG_CFLAGS = -std=c11 -Wall -Wextra -g -O0 -DDEBUG
DEBUG_CFLAGS += -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L

# Libraries
LIBS = -lm -lpthread -lopenblas -lfftw3f -ldl
INCLUDES = -I./include -I./src

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin
TEST_DIR = tests

# Source files
CORE_SOURCES = $(wildcard $(SRC_DIR)/kernel/*.c) \
               $(wildcard $(SRC_DIR)/wm/*.c) \
               $(wildcard $(SRC_DIR)/workspace/*.c) \
               $(wildcard $(SRC_DIR)/ltm/*.c) \
               $(wildcard $(SRC_DIR)/reason/*.c) \
               $(wildcard $(SRC_DIR)/self/*.c) \
               $(wildcard $(SRC_DIR)/io/*.c) \
               $(wildcard $(SRC_DIR)/utils/*.c) \
               $(wildcard $(SRC_DIR)/neural/*.c) \
               $(wildcard $(SRC_DIR)/cuda/*.c) \
               $(wildcard $(SRC_DIR)/reasoning/*.c) \
               $(wildcard $(SRC_DIR)/web/*.c)

DEMO_SOURCES = $(wildcard $(SRC_DIR)/demos/*.c)
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c)

# Object files
CORE_OBJECTS = $(CORE_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEMO_OBJECTS = $(DEMO_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
TEST_OBJECTS = $(TEST_SOURCES:$(TEST_DIR)/%.c=$(BUILD_DIR)/%.o)

# Targets
TARGETS = $(BIN_DIR)/ce_demo $(BIN_DIR)/ce_v1_1_demo $(BIN_DIR)/ce_test $(BIN_DIR)/ce_trace_viewer
LIBRARY = $(BUILD_DIR)/libconsciousness.a

.PHONY: all clean debug test install uninstall

all: $(TARGETS) $(LIBRARY)

debug: CFLAGS = $(DEBUG_CFLAGS)
debug: $(TARGETS) $(LIBRARY)

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/kernel $(BUILD_DIR)/wm $(BUILD_DIR)/workspace
	mkdir -p $(BUILD_DIR)/ltm $(BUILD_DIR)/reason $(BUILD_DIR)/self
	mkdir -p $(BUILD_DIR)/io $(BUILD_DIR)/utils $(BUILD_DIR)/demos

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Core library
$(LIBRARY): $(CORE_OBJECTS) | $(BUILD_DIR)
	ar rcs $@ $^

# Demo executables
$(BIN_DIR)/ce_demo: $(BUILD_DIR)/demos/basic_demo.o $(LIBRARY) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(BIN_DIR)/ce_v1_1_demo: $(BUILD_DIR)/demos/v1_1_demo.o $(LIBRARY) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Test executable
$(BIN_DIR)/ce_test: $(TEST_OBJECTS) $(LIBRARY) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Trace viewer
$(BIN_DIR)/ce_trace_viewer: $(SRC_DIR)/tools/trace_viewer.c $(LIBRARY) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBRARY) $(LIBS)

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Test target
test: $(BIN_DIR)/ce_test
	./$(BIN_DIR)/ce_test

# Install target
install: all
	@echo "Installing Consciousness Emulator..."
	@mkdir -p /usr/local/include/consciousness
	@mkdir -p /usr/local/lib
	@cp -r $(INCLUDE_DIR)/* /usr/local/include/consciousness/
	@cp $(LIBRARY) /usr/local/lib/
	@cp $(TARGETS) /usr/local/bin/
	@echo "Installation complete!"

# Uninstall target
uninstall:
	@echo "Uninstalling Consciousness Emulator..."
	@rm -rf /usr/local/include/consciousness
	@rm -f /usr/local/lib/libconsciousness.a
	@rm -f /usr/local/bin/ce_*
	@echo "Uninstallation complete!"

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Help target
help:
	@echo "Consciousness Emulator (CE) Build System"
	@echo "Author: AmirHosseinRasti"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build all targets (default)"
	@echo "  debug    - Build with debug flags"
	@echo "  test     - Run test suite"
	@echo "  install  - Install to system"
	@echo "  uninstall- Remove from system"
	@echo "  clean    - Remove build artifacts"
	@echo "  help     - Show this help"
