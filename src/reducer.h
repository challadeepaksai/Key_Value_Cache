#pragma once
#ifndef REDUCER_H_
#define REDUCER_H_

#include <concepts>
#include <functional>

namespace reducer {

/// Reducer: accumulate by addition.
template <std::regular T>
const std::function<void(T&, const T&)> sum = [](T& t1, const T& t2) {
  t1 += t2;
};

/// Reducer: keep the maximum.
template <std::totally_ordered T>
const std::function<void(T&, const T&)> max = [](T& t1, const T& t2) {
  if (t1 < t2) t1 = t2;
};

/// Reducer: keep the minimum.
template <std::totally_ordered T>
const std::function<void(T&, const T&)> min = [](T& t1, const T& t2) {
  if (t1 > t2) t1 = t2;
};

}  // namespace reducer

#endif  // REDUCER_H_
