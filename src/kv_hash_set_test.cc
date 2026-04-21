#include "kv_hash_set.h"

#include "gtest/gtest.h"
#include "reducer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

TEST(KVHashSetTest, Initialization) {
  kv_hash_set<std::string> s;
  EXPECT_EQ(s.get_n_keys(), 0u);
  EXPECT_EQ(s.size(), 0u);
}

TEST(KVHashSetTest, Reserve) {
  kv_hash_set<std::string> s;
  s.reserve(10);
  EXPECT_GE(s.get_n_buckets(), 10u);

  kv_hash_set<int> s2;
  for (int i = 0; i < 100; i++) {
    s2.add(i);
    EXPECT_EQ(s2.get_n_keys(), static_cast<std::size_t>(i + 1));
    EXPECT_GE(s2.get_n_buckets(), static_cast<std::size_t>(i + 1));
  }
  for (int i = 0; i < 100; i++) {
    EXPECT_TRUE(s2.has(i));
  }
}

TEST(KVHashSetTest, OneMillionReserve) {
  kv_hash_set<std::string> s;
  constexpr std::size_t N = 1000000;
  s.reserve(N);
  EXPECT_GE(s.get_n_buckets(), N);
}

TEST(KVHashSetLargeTest, HundredMillionsReserve) {
  kv_hash_set<std::string> s;
  constexpr std::size_t N = 100000000;
  s.reserve(N);
  EXPECT_GE(s.get_n_buckets(), N);
}

TEST(KVHashSetTest, Add) {
  kv_hash_set<std::string> s;
  s.add("aa");
  EXPECT_TRUE(s.has("aa"));
  s.add("aa");
  EXPECT_TRUE(s.has("aa"));

  s.add("bbb");
  EXPECT_TRUE(s.has("aa"));
  EXPECT_TRUE(s.has("bbb"));
  EXPECT_FALSE(s.has("not_exist_key"));
}

TEST(KVHashSetTest, Contains) {
  kv_hash_set<std::string> s;
  s.add("hello");
  EXPECT_TRUE(s.contains("hello"));
  EXPECT_FALSE(s.contains("world"));
}

TEST(KVHashSetLargeTest, TenMillionsInsertWithAutoRehash) {
  kv_hash_set<int> s;
  constexpr int N = 10000000;

#ifdef _OPENMP
  omp_set_nested(1);
#pragma omp parallel for
#endif
  for (int i = 0; i < N; i++) {
    s.add(i);
  }
  EXPECT_EQ(s.get_n_keys(), static_cast<std::size_t>(N));
  EXPECT_GE(s.get_n_buckets(), static_cast<std::size_t>(N));
}

TEST(KVHashSetTest, Remove) {
  kv_hash_set<std::string> s;
  s.add("aa");
  s.add("bbb");
  s.remove("aa");
  EXPECT_FALSE(s.has("aa"));
  EXPECT_TRUE(s.has("bbb"));
  EXPECT_EQ(s.get_n_keys(), 1u);

  s.remove("not_exist_key");
  EXPECT_EQ(s.get_n_keys(), 1u);

  s.remove("bbb");
  EXPECT_FALSE(s.has("aa"));
  EXPECT_FALSE(s.has("bbb"));
  EXPECT_EQ(s.get_n_keys(), 0u);
}

TEST(KVHashSetTest, Apply) {
  kv_hash_set<std::string> s;
  s.add("aa");
  s.add("bbb");
  int count = 0;

  s.apply([&](const auto& key) {
    if (key.front() == 'a') count++;
  });
  EXPECT_EQ(count, 1);
}

TEST(KVHashSetTest, MapReduce) {
  kv_hash_set<std::string> s;
  s.add("aa");
  s.add("ab");
  s.add("ac");
  s.add("ad");
  s.add("ae");
  s.add("ba");
  s.add("bb");

  const auto& a_to_one = [&](const std::string& key) {
    return key.front() == 'a' ? 1 : 0;
  };
  const int count = s.map_reduce<int>(a_to_one, reducer::sum<int>, 0);
  EXPECT_EQ(count, 5);
}

TEST(KVHashSetLargeTest, TenMillionsMapReduce) {
  kv_hash_set<int> s;
  constexpr int N = 10000000;
  s.reserve(N);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; i++) {
    s.add(i);
  }
  const auto& mapper = [&](const int key) { return key; };
  const auto& result = s.map_reduce<int>(mapper, reducer::max<int>, 0);
  EXPECT_EQ(result, N - 1);
}

TEST(KVHashSetTest, Clear) {
  kv_hash_set<std::string> s;
  s.add("aa");
  s.add("bbb");
  s.clear();
  EXPECT_FALSE(s.has("aa"));
  EXPECT_FALSE(s.has("bbb"));
  EXPECT_EQ(s.get_n_keys(), 0u);
}
