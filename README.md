# Key_Value_hash

A high-performance, thread-safe C++20 hash map and hash set library that **outperforms** traditional mutex-wrapped `std::unordered_map` in concurrent workloads through fine-grained segment locking and OpenMP-accelerated bulk operations.

## Features

- **Thread-safe** concurrent access via fine-grained `std::shared_mutex` segment locking
- **Parallel** rehashing, clearing, and map-reduce (OpenMP)
- **Modern C++20** — concepts, `[[nodiscard]]`, RAII locks, `std::make_unique`
- **Get-and-set in one shot** — update values with a single hash lookup
- **Header-only** — just include and compile

## Quick Start

```cpp
#include "kv_hash_map.h"
#include "reducer.h"
#include <string>

int main() {
  kv_hash_map<std::string, int> m;

  // Insert / update
  m.set("a", 0);
  m.set("b", 1);
  m.set("c", 2);
  m.get_copy_or_default("b", 0);  // → 1

  // In-place update with a single lookup
  m.set("b", [](auto& v) { v++; });
  m.get_copy_or_default("b", 0);  // → 2

  // STL-style helpers
  m.contains("a");  // → true
  m.size();          // → 3

  // Parallel map-reduce
  auto square = [](const std::string&, int v) { return v * v; };
  int sum = m.map_reduce<int>(square, reducer::sum<int>, 0);  // → 8
}
```

## Prerequisites

- **C++20** compatible compiler (Apple Clang 15+, GCC 11+)
- **OpenMP** runtime (`libomp`)

```bash
# macOS — install libomp via Homebrew
brew install libomp
```

## Building & Running

```bash
# Clone and enter the project
cd Key_Value_hash

# Run unit tests (excludes large/stress tests)
make test

# Run all tests including large/stress tests
make all_tests

# Build and run the benchmark
make bench_all

# Clean build artifacts
make clean
```

## Benchmark Results

### Concurrent Read/Write Scaling

**500K ops/thread, 200K key range, Apple M-series (10 cores)**

The benchmark compares `kv_hash_map` (fine-grained segment locking) against a mutex-wrapped `std::unordered_map` (single global lock) as thread count increases:

| Threads | `kv_hash_map` (ms) | `mutex<unordered_map>` (ms) | Speedup |
|---------|--------------------|-----------------------------|---------|
| 1       | 38.08              | 15.50                       | 0.41x   |
| 2       | 78.21              | 47.01                       | 0.60x   |
| 4       | 137.58             | 107.13                      | 0.78x   |
| **8**   | **272.56**         | 301.12                      | **1.10x** |
| **10**  | **329.74**         | 460.56                      | **1.40x** |

**Key takeaway:** At higher thread counts (8+), `kv_hash_map` **outperforms** the traditional approach by **up to 1.4x** because its fine-grained segment locking avoids the contention bottleneck of a single global mutex. The more threads, the bigger the advantage.

## Project Structure

```
src/
├── kv_hash_map.h          # Thread-safe hash map (header-only)
├── kv_hash_set.h          # Thread-safe hash set (header-only)
├── reducer.h              # Built-in reducers (sum, max, min)
├── kv_hash_map_test.cc    # Hash map unit tests
├── kv_hash_set_test.cc    # Hash set unit tests
├── reducer_test.cc        # Reducer unit tests
└── googletest/            # Google Test framework (vendored)
benchmarks/
└── bench_concurrent_rw.cc # Concurrent read/write scaling benchmark
Makefile                   # Build system
```
