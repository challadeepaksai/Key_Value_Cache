# Default compiler – Apple Clang on macOS.
ifndef CXX
CXX := clang++
endif

# Homebrew libomp paths (Apple Silicon / Intel).
OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
OMP_INCLUDE := -I$(OMP_PREFIX)/include
OMP_LIBDIR  := -L$(OMP_PREFIX)/lib

# C++20, OpenMP via Xpreprocessor (Apple Clang), optimised with debug info.
CXXFLAGS := -std=c++20 -Wall -Wextra -O3 -Xpreprocessor -fopenmp $(OMP_INCLUDE) -g
LDLIBS   := $(OMP_LIBDIR) -lomp

SRC_DIR  := src
OBJ_DIR  := build
TEST_EXE := test.out
BENCH_DIR := benchmarks

# ---------- Source discovery ----------
TESTS    := $(wildcard $(SRC_DIR)/kv_*_test.cc $(SRC_DIR)/reducer_test.cc)
HEADERS  := $(wildcard $(SRC_DIR)/*.h)
TEST_OBJS := $(TESTS:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)

# ---------- GoogleTest ----------
GTEST_DIR      := $(SRC_DIR)/googletest/googletest
GTEST_CXXFLAGS := $(CXXFLAGS) -isystem $(GTEST_DIR)/include -pthread
GTEST_HEADERS  := $(GTEST_DIR)/include/gtest/*.h \
                  $(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS     := $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)
GTEST_MAIN     := $(OBJ_DIR)/gtest_main.a

# ---------- Benchmark ----------
BENCH_SRC := $(BENCH_DIR)/bench_concurrent_rw.cc
BENCH_BIN := $(OBJ_DIR)/bench_concurrent_rw

.PHONY: all test all_tests clean bench_all

all: test

# Run unit tests (skip large tests by default).
test: $(TEST_EXE)
	./$(TEST_EXE) --gtest_filter=-*LargeTest.*

# Run all tests including large / stress tests.
all_tests: $(TEST_EXE)
	./$(TEST_EXE)

# Build and run the benchmark.
bench_all: $(BENCH_BIN)
	@echo "==========================================="
	@echo "  Running Concurrent R/W Benchmark"
	@echo "==========================================="
	./$(BENCH_BIN)

clean:
	rm -rf $(OBJ_DIR)
	rm -f  ./$(TEST_EXE)

# ---------- Test linking ----------
$(TEST_EXE): $(TEST_OBJS) $(GTEST_MAIN) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(TEST_OBJS) $(GTEST_MAIN) -o $(TEST_EXE) $(LDLIBS) -lpthread

$(TEST_OBJS): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc $(HEADERS)
	mkdir -p $(@D) && $(CXX) $(GTEST_CXXFLAGS) -c $< -o $@

# ---------- GoogleTest static lib ----------
$(GTEST_MAIN): $(OBJ_DIR)/gtest-all.o $(OBJ_DIR)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

$(OBJ_DIR)/gtest-all.o: $(GTEST_SRCS)
	mkdir -p $(@D) && $(CXX) -I$(GTEST_DIR) $(GTEST_CXXFLAGS) -c \
		$(GTEST_DIR)/src/gtest-all.cc -o $@

$(OBJ_DIR)/gtest_main.o: $(GTEST_SRCS)
	mkdir -p $(@D) && $(CXX) -I$(GTEST_DIR) $(GTEST_CXXFLAGS) -c \
		$(GTEST_DIR)/src/gtest_main.cc -o $@

# ---------- Benchmark ----------
$(BENCH_BIN): $(BENCH_SRC) $(HEADERS)
	mkdir -p $(@D) && $(CXX) $(CXXFLAGS) -I$(SRC_DIR) $< -o $@ $(LDLIBS) -lpthread
