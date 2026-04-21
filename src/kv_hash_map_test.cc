#include "kv_hash_map.h"

#include "gtest/gtest.h"
#include "reducer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

TEST(KVHashMapTest, Initialization) {
  kv_hash_map<std::string, int> m;
  EXPECT_EQ(m.get_n_keys(), 0u);
  EXPECT_EQ(m.size(), 0u);
}

TEST(KVHashMapTest, Reserve) {
  kv_hash_map<std::string, int> m;
  m.reserve(10);
  EXPECT_GE(m.get_n_buckets(), 10u);

  kv_hash_map<int, int> m2;
  for (int i = 0; i < 100; i++) {
    m2.set(i, i * i);
    EXPECT_EQ(m2.get_n_keys(), static_cast<std::size_t>(i + 1));
    EXPECT_GE(m2.get_n_buckets(), static_cast<std::size_t>(i + 1));
  }
  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(m2.get_copy_or_default(i, 0), i * i);
  }
}

TEST(KVHashMapTest, OneMillionReserve) {
  kv_hash_map<std::string, int> m;
  constexpr std::size_t LARGE_N = 1000000;
  m.reserve(LARGE_N);
  EXPECT_GE(m.get_n_buckets(), LARGE_N);
}

TEST(KVHashMapLargeTest, HundredMillionsReserve) {
  kv_hash_map<std::string, int> m;
  constexpr std::size_t LARGE_N = 100000000;
  m.reserve(LARGE_N);
  EXPECT_GE(m.get_n_buckets(), LARGE_N);
}

TEST(KVHashMapTest, Set) {
  kv_hash_map<std::string, int> m;
  m.set("aa", 0);
  EXPECT_EQ(m.get_copy_or_default("aa", 0), 0);
  m.set("aa", 1);
  EXPECT_EQ(m.get_copy_or_default("aa", 0), 1);

  const auto& inc = [&](auto& value) { value++; };
  m.set("aa", inc);
  EXPECT_EQ(m.get_copy_or_default("aa", 0), 2);

  kv_hash_map<std::string, std::string> m2;
  m2.set("cc", [&](auto& value) { value = value + "x"; });
  EXPECT_EQ(m2.get_copy_or_default("cc", ""), "x");

  m.set("aa", inc, 0);
  EXPECT_EQ(m.get_copy_or_default("aa", 0), 3);
  m.set("bbb", inc, 5);
  EXPECT_EQ(m.get_copy_or_default("bbb", 0), 6);
}

TEST(KVHashMapLargeTest, TenMillionsInsertWithAutoRehash) {
  kv_hash_map<int, int> m;
  constexpr int N = 10000000;

#ifdef _OPENMP
  omp_set_nested(1);
#pragma omp parallel for
#endif
  for (int i = 0; i < N; i++) {
    m.set(i, i);
  }
  EXPECT_EQ(m.get_n_keys(), static_cast<std::size_t>(N));
  EXPECT_GE(m.get_n_buckets(), static_cast<std::size_t>(N));
}

TEST(KVHashMapTest, Unset) {
  kv_hash_map<std::string, int> m;
  m.set("aa", 1);
  m.set("bbb", 2);
  m.unset("aa");
  EXPECT_FALSE(m.has("aa"));
  EXPECT_TRUE(m.has("bbb"));
  EXPECT_EQ(m.get_n_keys(), 1u);

  m.unset("not_exist_key");
  EXPECT_EQ(m.get_n_keys(), 1u);

  m.unset("bbb");
  EXPECT_FALSE(m.has("aa"));
  EXPECT_FALSE(m.has("bbb"));
  EXPECT_EQ(m.get_n_keys(), 0u);
}

TEST(KVHashMapTest, Contains) {
  kv_hash_map<std::string, int> m;
  m.set("hello", 42);
  EXPECT_TRUE(m.contains("hello"));
  EXPECT_FALSE(m.contains("world"));
}

TEST(KVHashMapTest, Map) {
  kv_hash_map<std::string, int> m;
  const auto& cubic = [&](const int value) { return value * value * value; };
  m.set("aa", 5);
  EXPECT_EQ(m.map<int>("aa", cubic, 0), 125);
  EXPECT_EQ(m.map<int>("not_exist_key", cubic, 3), 3);
}

TEST(KVHashMapTest, Apply) {
  kv_hash_map<std::string, int> m;
  m.set("aa", 5);
  m.set("bbb", 10);
  int sum = 0;

  m.apply("aa", [&](const auto& value) { sum += value; });
  EXPECT_EQ(sum, 5);

  m.apply([&](const auto& key, const auto& value) {
    if (key.front() == 'b') sum += value;
  });
  EXPECT_EQ(sum, 15);
}

TEST(KVHashMapTest, MapReduce) {
  kv_hash_map<std::string, double> m;
  m.set("aa", 1.1);
  m.set("ab", 2.2);
  m.set("ac", 3.3);
  m.set("ad", 4.4);
  m.set("ae", 5.5);
  m.set("ba", 6.6);
  m.set("bb", 7.7);

  const auto& a_to_one = [&](const std::string& key, const auto value) {
    (void)value;
    return key.front() == 'a' ? 1 : 0;
  };
  const int count = m.map_reduce<int>(a_to_one, reducer::sum<int>, 0);
  EXPECT_EQ(count, 5);
}

TEST(KVHashMapLargeTest, TenMillionsMapReduce) {
  kv_hash_map<int, int> m;
  constexpr int N = 10000000;
  m.reserve(N);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; i++) {
    m.set(i, i);
  }
  const auto& mapper = [&](const int key, const int value) {
    (void)key;
    return value;
  };
  const auto& result = m.map_reduce<int>(mapper, reducer::max<int>, 0);
  EXPECT_EQ(result, N - 1);
}

TEST(KVHashMapTest, Clear) {
  kv_hash_map<std::string, int> m;
  m.set("aa", 1);
  m.set("bbb", 2);
  m.clear();
  EXPECT_FALSE(m.has("aa"));
  EXPECT_FALSE(m.has("bbb"));
  EXPECT_EQ(m.get_n_keys(), 0u);
}
