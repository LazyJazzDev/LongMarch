#pragma once
#include "grassland/util/util_util.h"

namespace grassland {

template <class T>
LM_DEVICE_FUNC int BinarySearch(const T *keys, int num_key, T key) {
  int l = 0, r = num_key - 1;
  while (l < r) {
    int m = (l + r) >> 1;
    if (keys[m] < key) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  if (key < keys[l] || keys[r] < key) {
    return -1;
  }
  return l;
}

}  // namespace grassland
