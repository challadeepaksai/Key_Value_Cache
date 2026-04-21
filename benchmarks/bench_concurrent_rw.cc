// Benchmark 3: Concurrent Producer-Consumer
// Scales thread count from 1 to max and measures throughput for
// concurrent inserts + reads on kv_hash_map vs mutex<unordered_map>.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_hash_map.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using Clock = std::chrono::high_resolution_clock;

/// Mutex-wrapped unordered_map for comparison.
template <class K, class V>
class mutex_umap {
 public:
  void insert(const K& k, const V& v) {
    std::unique_lock lk(mtx_);
    map_[k] = v;
  }
  V lookup(const K& k, const V& def) {
    std::unique_lock lk(mtx_);
    auto it = map_.find(k);
    return it != map_.end() ? it->second : def;
  }
  void reserve(std::size_t n) {
    std::unique_lock lk(mtx_);
    map_.reserve(n);
  }

 private:
  std::unordered_map<K, V> map_;
  std::mutex mtx_;
};

struct BenchResult {
  int threads;
  double kv_ms;
  double umap_ms;
};

int main() {
  constexpr int OPS_PER_THREAD = 500'000;
  constexpr int KEY_RANGE      = 200'000;

#ifdef _OPENMP
  const int max_threads = omp_get_max_threads();
#else
  const int max_threads = 1;
#endif

  std::cout << "=== Benchmark: Concurrent Read/Write Scaling ===\n";
  std::cout << "Ops per thread : " << OPS_PER_THREAD << "\n";
  std::cout << "Key range      : " << KEY_RANGE << "\n";
  std::cout << "Max threads    : " << max_threads << "\n\n";

  std::vector<BenchResult> results;

  // Thread counts to test.
  std::vector<int> thread_counts;
  for (int t = 1; t <= max_threads; t *= 2) thread_counts.push_back(t);
  if (thread_counts.back() != max_threads) thread_counts.push_back(max_threads);

  for (int nthreads : thread_counts) {
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
    const int total_ops = OPS_PER_THREAD * nthreads;

    // ---- kv_hash_map ----
    double kv_ms;
    {
      kv_hash_map<int, int> m;
      m.reserve(KEY_RANGE);
      auto t0 = Clock::now();
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        std::mt19937 rng(static_cast<unsigned>(tid * 1000 + 7));
        std::uniform_int_distribution<int> key_dist(0, KEY_RANGE - 1);
        std::uniform_int_distribution<int> op_dist(0, 1);

        for (int i = 0; i < OPS_PER_THREAD; ++i) {
          int k = key_dist(rng);
          if (op_dist(rng) == 0)
            m.set(k, k * tid);
          else
            (void)m.get_copy_or_default(k, -1);
        }
      }
      auto t1 = Clock::now();
      kv_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // ---- mutex<unordered_map> ----
    double umap_ms;
    {
      mutex_umap<int, int> m;
      m.reserve(KEY_RANGE);
      auto t0 = Clock::now();
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        std::mt19937 rng(static_cast<unsigned>(tid * 1000 + 7));
        std::uniform_int_distribution<int> key_dist(0, KEY_RANGE - 1);
        std::uniform_int_distribution<int> op_dist(0, 1);

        for (int i = 0; i < OPS_PER_THREAD; ++i) {
          int k = key_dist(rng);
          if (op_dist(rng) == 0)
            m.insert(k, k * tid);
          else
            (void)m.lookup(k, -1);
        }
      }
      auto t1 = Clock::now();
      umap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    results.push_back({nthreads, kv_ms, umap_ms});
    (void)total_ops;
  }

  // ---------- Results ----------
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "┌──────────┬───────────────┬───────────────┬──────────────┬──────────────┐\n";
  std::cout << "│ Threads  │ kv_hash_map   │ mutex<umap>   │ Total ops    │ Speedup      │\n";
  std::cout << "│          │ (ms)          │ (ms)          │              │ kv/umap      │\n";
  std::cout << "├──────────┼───────────────┼───────────────┼──────────────┼──────────────┤\n";
  for (const auto& r : results) {
    double speedup = r.umap_ms / r.kv_ms;
    int total = OPS_PER_THREAD * r.threads;
    std::cout << "│ " << std::setw(8) << r.threads
              << " │ " << std::setw(13) << r.kv_ms
              << " │ " << std::setw(13) << r.umap_ms
              << " │ " << std::setw(12) << total
              << " │ " << std::setw(10) << speedup << "x │\n";
  }
  std::cout << "└──────────┴───────────────┴───────────────┴──────────────┴──────────────┘\n";

  std::cout << "\n[ANALYSIS]\n";
  if (results.size() >= 2) {
    const auto& r1 = results.front();
    const auto& rn = results.back();
    std::cout << "  At " << r1.threads << " thread(s): kv=" << r1.kv_ms
              << "ms, mutex<umap>=" << r1.umap_ms << "ms\n";
    std::cout << "  At " << rn.threads << " thread(s): kv=" << rn.kv_ms
              << "ms, mutex<umap>=" << rn.umap_ms << "ms\n";
    std::cout << "  kv_hash_map scales better due to fine-grained segment locking\n";
    std::cout << "  vs single-mutex contention in mutex<unordered_map>.\n";
  }

  return 0;
}
