#include "reducer.h"

#include "gtest/gtest.h"
#include "kv_hash_map.h"

TEST(ReducerTest, ReduceSum) {
  kv_hash_map<int, int> m;
  for (int i = 0; i < 100; i++) m.set(i, i);
  const auto& get_value = [](const int key, const int value) {
    (void)key;
    return value;
  };
  EXPECT_EQ(m.map_reduce<int>(get_value, reducer::sum<int>, 0), 4950);
}

TEST(ReducerTest, ReduceMax) {
  kv_hash_map<int, int> m;
  for (int i = 0; i < 100; i++) m.set(i, i);
  const auto& get_value = [](const int key, const int value) {
    (void)key;
    return value;
  };
  EXPECT_EQ(m.map_reduce<int>(get_value, reducer::max<int>, 0), 99);
}

TEST(ReducerTest, ReduceMin) {
  kv_hash_map<int, int> m;
  for (int i = 0; i < 100; i++) m.set(i, i);
  const auto& get_value = [](const int key, const int value) {
    (void)key;
    return value;
  };
  EXPECT_EQ(m.map_reduce<int>(get_value, reducer::min<int>, 0), 0);
}