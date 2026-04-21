#pragma once
#ifndef KV_HASH_MAP_H_
#define KV_HASH_MAP_H_

#include <array>
#include <concepts>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------- Concepts (C++20) ----------

template <class H, class K>
concept Hashable = requires(H h, K k) {
  { h(k) } -> std::convertible_to<std::size_t>;
};

template <class K>
concept EqualityComparable = requires(K a, K b) {
  { a == b } -> std::convertible_to<bool>;
};

// ---------- kv_hash_map ----------

/// A high-performance, thread-safe hash map.
/// Uses fine-grained segment locking (std::shared_mutex) for concurrent access
/// and OpenMP for parallel bulk operations (rehash, clear, map-reduce).
template <EqualityComparable K, class V, Hashable<K> H = std::hash<K>>
class kv_hash_map {
 public:
  kv_hash_map();
  ~kv_hash_map();

  // Non-copyable, movable.
  kv_hash_map(const kv_hash_map&) = delete;
  kv_hash_map& operator=(const kv_hash_map&) = delete;
  kv_hash_map(kv_hash_map&&) noexcept = default;
  kv_hash_map& operator=(kv_hash_map&&) noexcept = default;

  /// Reserve at least @p n_buckets buckets.
  void reserve(std::size_t n_buckets) {
    const std::size_t n_rehashing_buckets = get_n_rehashing_buckets(n_buckets);
    rehash(n_rehashing_buckets);
  }

  /// @return current number of buckets.
  [[nodiscard]] std::size_t get_n_buckets() const { return n_buckets_; }

  /// @return current load factor.
  [[nodiscard]] double get_load_factor() const {
    return static_cast<double>(n_keys_) / n_buckets_;
  }

  /// @return max load factor threshold.
  [[nodiscard]] double get_max_load_factor() const { return max_load_factor_; }

  /// Set the max load factor.
  void set_max_load_factor(double mlf) { max_load_factor_ = mlf; }

  /// @return number of stored keys.
  [[nodiscard]] std::size_t get_n_keys() const { return n_keys_; }

  /// STL-compatible alias for get_n_keys().
  [[nodiscard]] std::size_t size() const { return n_keys_; }

  /// @return true if the map contains @p key.
  [[nodiscard]] bool contains(const K& key) { return has(key); }

  // ---- Mutators ----

  /// Set @p key to @p value (insert or overwrite).
  void set(const K& key, const V& value);

  /// Apply @p setter to the value of @p key. If the key does not exist, it is
  /// default-constructed first.
  void set(const K& key, const std::function<void(V&)>& setter);

  /// Apply @p setter to the value of @p key. If the key does not exist, it is
  /// initialised to @p default_value first.
  void set(const K& key, const std::function<void(V&)>& setter,
           const V& default_value);

  /// Remove @p key. No-op if absent.
  void unset(const K& key);

  // ---- Observers ----

  /// @return true if @p key exists.
  bool has(const K& key);

  /// @return copy of value for @p key, or @p default_value if absent.
  V get_copy_or_default(const K& key, const V& default_value);

  /// Map the value of @p key through @p mapper, or return @p default_value.
  template <class W>
  W map(const K& key, const std::function<W(const V&)>& mapper,
        const W& default_value);

  /// Parallel map-reduce over all keys.
  template <class W>
  W map_reduce(const std::function<W(const K&, const V&)>& mapper,
               const std::function<void(W&, const W&)>& reducer,
               const W& default_value);

  /// Apply @p handler to the value of @p key (if it exists).
  void apply(const K& key, const std::function<void(const V&)>& handler);

  /// Apply @p handler to every (key, value) pair.
  void apply(const std::function<void(const K&, const V&)>& handler);

  /// Remove all keys (parallel).
  void clear();

 private:
  std::size_t n_keys_{0};
  std::size_t n_buckets_{N_INITIAL_BUCKETS};
  double max_load_factor_{DEFAULT_MAX_LOAD_FACTOR};

  std::size_t n_threads_{1};
  std::size_t n_segments_{N_SEGMENTS_PER_THREAD};

  H hasher_{};

  std::vector<std::shared_mutex> segment_mutexes_;
  std::vector<std::shared_mutex> rehashing_segment_mutexes_;

  static constexpr std::size_t N_INITIAL_BUCKETS = 11;
  static constexpr std::size_t N_SEGMENTS_PER_THREAD = 7;
  static constexpr double DEFAULT_MAX_LOAD_FACTOR = 1.0;

  struct hash_node {
    K key;
    V value;
    std::unique_ptr<hash_node> next;
    hash_node(const K& k, const V& v) : key(k), value(v) {}
  };

  std::vector<std::unique_ptr<hash_node>> buckets_;

  void rehash() { reserve(static_cast<std::size_t>(n_keys_ / max_load_factor_)); }
  void rehash(std::size_t n_rehashing_buckets);

  [[nodiscard]] std::size_t get_n_rehashing_buckets(std::size_t n) const;

  void hash_node_apply(const K& key,
                       const std::function<void(std::unique_ptr<hash_node>&)>& handler);
  void hash_node_apply(const std::function<void(std::unique_ptr<hash_node>&)>& handler);

  void hash_node_apply_recursive(
      std::unique_ptr<hash_node>& node, const K& key,
      const std::function<void(std::unique_ptr<hash_node>&)>& handler);
  void hash_node_apply_recursive(
      std::unique_ptr<hash_node>& node,
      const std::function<void(std::unique_ptr<hash_node>&)>& handler);

  void lock_all_segments();
  void unlock_all_segments();
};

// =====================================================================
//  Implementation
// =====================================================================

template <EqualityComparable K, class V, Hashable<K> H>
kv_hash_map<K, V, H>::kv_hash_map() {
  n_keys_ = 0;
  n_buckets_ = N_INITIAL_BUCKETS;
  buckets_.resize(n_buckets_);
  max_load_factor_ = DEFAULT_MAX_LOAD_FACTOR;

#ifdef _OPENMP
  n_threads_ = static_cast<std::size_t>(omp_get_max_threads());
#else
  n_threads_ = 1;
#endif
  n_segments_ = n_threads_ * N_SEGMENTS_PER_THREAD;
  segment_mutexes_ = std::vector<std::shared_mutex>(n_segments_);
  rehashing_segment_mutexes_ = std::vector<std::shared_mutex>(n_segments_);
}

template <EqualityComparable K, class V, Hashable<K> H>
kv_hash_map<K, V, H>::~kv_hash_map() {
  // RAII handles mutex destruction automatically.
  clear();
}

// ---- rehash ----

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::rehash(std::size_t n_rehashing_buckets) {
  lock_all_segments();

  if (n_buckets_ >= n_rehashing_buckets) {
    unlock_all_segments();
    return;
  }

  std::vector<std::unique_ptr<hash_node>> rehashing_buckets(n_rehashing_buckets);
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    const auto& rehashing_node_handler = [&](std::unique_ptr<hash_node>& rnode) {
      rnode = std::move(node);
      rnode->next.reset();
    };
    const K& key = node->key;
    const std::size_t hash_value = hasher_(key);
    const std::size_t bucket_id = hash_value % n_rehashing_buckets;
    const std::size_t segment_id = bucket_id % n_segments_;
    {
      std::unique_lock lock(rehashing_segment_mutexes_[segment_id]);
      hash_node_apply_recursive(rehashing_buckets[bucket_id], key,
                                rehashing_node_handler);
    }
  };

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t i = 0; i < n_buckets_; i++) {
    hash_node_apply_recursive(buckets_[i], node_handler);
  }

  buckets_ = std::move(rehashing_buckets);
  n_buckets_ = n_rehashing_buckets;
  unlock_all_segments();
}

template <EqualityComparable K, class V, Hashable<K> H>
std::size_t kv_hash_map<K, V, H>::get_n_rehashing_buckets(std::size_t n_buckets_in) const {
  constexpr std::size_t PRIME_NUMBERS[] = {
      11,    17,    29,    47,    79,     127,    211,
      337,   547,   887,   1433,  2311,   3739,   6053,
      9791,  15858, 25667, 41539, 67213,  104729};
  constexpr std::size_t N_PRIME_NUMBERS = sizeof(PRIME_NUMBERS) / sizeof(std::size_t);
  constexpr std::size_t LAST_PRIME = PRIME_NUMBERS[N_PRIME_NUMBERS - 1];
  constexpr std::size_t DIVISION_FACTOR = 15858;

  std::size_t remaining = n_buckets_in;
  std::size_t result = 1;
  for (int i = 0; i < 3; ++i) {
    if (remaining > LAST_PRIME) {
      remaining /= DIVISION_FACTOR;
      result *= DIVISION_FACTOR;
    }
  }
  if (remaining > LAST_PRIME)
    throw std::invalid_argument("n_buckets too large");

  std::size_t lo = 0, hi = N_PRIME_NUMBERS - 1;
  while (lo < hi) {
    std::size_t mid = (lo + hi) / 2;
    if (PRIME_NUMBERS[mid] < remaining)
      lo = mid + 1;
    else
      hi = mid;
  }
  result *= PRIME_NUMBERS[lo];
  return result;
}

// ---- set ----

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::set(const K& key, const V& value) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (!node) {
      node = std::make_unique<hash_node>(key, value);
#ifdef _OPENMP
#pragma omp atomic
#endif
      n_keys_++;
    } else {
      node->value = value;
    }
  };
  hash_node_apply(key, node_handler);
  if (n_keys_ >= static_cast<std::size_t>(n_buckets_ * max_load_factor_))
    rehash();
}

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::set(const K& key,
                                const std::function<void(V&)>& setter) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (!node) {
      node = std::make_unique<hash_node>(key, V());
      setter(node->value);
#ifdef _OPENMP
#pragma omp atomic
#endif
      n_keys_++;
    } else {
      setter(node->value);
    }
  };
  hash_node_apply(key, node_handler);
  if (n_keys_ >= static_cast<std::size_t>(n_buckets_ * max_load_factor_))
    rehash();
}

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::set(const K& key,
                                const std::function<void(V&)>& setter,
                                const V& default_value) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (!node) {
      V value(default_value);
      setter(value);
      node = std::make_unique<hash_node>(key, value);
#ifdef _OPENMP
#pragma omp atomic
#endif
      n_keys_++;
    } else {
      setter(node->value);
    }
  };
  hash_node_apply(key, node_handler);
  if (n_keys_ >= static_cast<std::size_t>(n_buckets_ * max_load_factor_))
    rehash();
}

// ---- unset ----

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::unset(const K& key) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (node) {
      node = std::move(node->next);
#ifdef _OPENMP
#pragma omp atomic
#endif
      n_keys_--;
    }
  };
  hash_node_apply(key, node_handler);
}

// ---- has / contains ----

template <EqualityComparable K, class V, Hashable<K> H>
bool kv_hash_map<K, V, H>::has(const K& key) {
  bool found = false;
  const auto& node_handler = [&](const std::unique_ptr<hash_node>& node) {
    if (node) found = true;
  };
  hash_node_apply(key, node_handler);
  return found;
}

// ---- get_copy_or_default ----

template <EqualityComparable K, class V, Hashable<K> H>
V kv_hash_map<K, V, H>::get_copy_or_default(const K& key,
                                              const V& default_value) {
  V value(default_value);
  const auto& node_handler = [&](const std::unique_ptr<hash_node>& node) {
    if (node) value = node->value;
  };
  hash_node_apply(key, node_handler);
  return value;
}

// ---- map ----

template <EqualityComparable K, class V, Hashable<K> H>
template <class W>
W kv_hash_map<K, V, H>::map(const K& key,
                              const std::function<W(const V&)>& mapper,
                              const W& default_value) {
  W mapped(default_value);
  const auto& node_handler = [&](const std::unique_ptr<hash_node>& node) {
    if (node) mapped = mapper(node->value);
  };
  hash_node_apply(key, node_handler);
  return mapped;
}

// ---- map_reduce ----

template <EqualityComparable K, class V, Hashable<K> H>
template <class W>
W kv_hash_map<K, V, H>::map_reduce(
    const std::function<W(const K&, const V&)>& mapper,
    const std::function<void(W&, const W&)>& reducer,
    const W& default_value) {
  std::vector<W> thread_values(n_threads_, default_value);
  W result = default_value;
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
#ifdef _OPENMP
    const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
#else
    const std::size_t tid = 0;
#endif
    const W& mapped = mapper(node->key, node->value);
    reducer(thread_values[tid], mapped);
  };
  hash_node_apply(node_handler);
  for (const auto& v : thread_values) reducer(result, v);
  return result;
}

// ---- apply ----

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::apply(const K& key,
                                   const std::function<void(const V&)>& handler) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (node) handler(node->value);
  };
  hash_node_apply(key, node_handler);
}

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::apply(
    const std::function<void(const K&, const V&)>& handler) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    handler(node->key, node->value);
  };
  hash_node_apply(node_handler);
}

// ---- clear ----

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::clear() {
  lock_all_segments();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t i = 0; i < n_buckets_; i++) {
    buckets_[i].reset();
  }

  buckets_.resize(N_INITIAL_BUCKETS);
  for (auto& b : buckets_) b.reset();
  n_keys_ = 0;
  unlock_all_segments();
}

// ---- Internal: hash_node_apply (single key) ----

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::hash_node_apply(
    const K& key,
    const std::function<void(std::unique_ptr<hash_node>&)>& handler) {
  const std::size_t hash_value = hasher_(key);
  bool applied = false;
  while (!applied) {
    const std::size_t snap = n_buckets_;
    const std::size_t bucket_id = hash_value % snap;
    const std::size_t segment_id = bucket_id % n_segments_;
    std::unique_lock lock(segment_mutexes_[segment_id]);
    if (snap != n_buckets_) continue;
    hash_node_apply_recursive(buckets_[bucket_id], key, handler);
    applied = true;
  }
}

// ---- Internal: hash_node_apply (all nodes, parallel) ----

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::hash_node_apply(
    const std::function<void(std::unique_ptr<hash_node>&)>& handler) {
  lock_all_segments();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t i = 0; i < n_buckets_; i++) {
    hash_node_apply_recursive(buckets_[i], handler);
  }
  unlock_all_segments();
}

// ---- Internal: recursive helpers ----

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::hash_node_apply_recursive(
    std::unique_ptr<hash_node>& node, const K& key,
    const std::function<void(std::unique_ptr<hash_node>&)>& handler) {
  if (node) {
    if (node->key == key)
      handler(node);
    else
      hash_node_apply_recursive(node->next, key, handler);
  } else {
    handler(node);
  }
}

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::hash_node_apply_recursive(
    std::unique_ptr<hash_node>& node,
    const std::function<void(std::unique_ptr<hash_node>&)>& handler) {
  if (node) {
    hash_node_apply_recursive(node->next, handler);
    handler(node);
  }
}

// ---- lock helpers ----

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::lock_all_segments() {
  for (auto& m : segment_mutexes_) m.lock();
}

template <EqualityComparable K, class V, Hashable<K> H>
void kv_hash_map<K, V, H>::unlock_all_segments() {
  for (auto& m : segment_mutexes_) m.unlock();
}

#endif  // KV_HASH_MAP_H_
