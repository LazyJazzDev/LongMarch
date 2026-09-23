#pragma once

#include <cmath>
#include <cstdint>

#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "vector"

template <class Ty>
Ty Mix(const Ty &v0, const Ty &v1, float alpha) {
  return v0 * (1.0f - alpha) + v1 * alpha;
}

float PowerInterpolation(float x, float index);

float CosineInterpolation(float x);

template <class Ty>
class MixValue {
 public:
  explicit MixValue(const std::vector<Ty> &values = {}) : values_(values) {
  }

  [[nodiscard]] Ty GetValue(float alpha) const {
    float index;
    alpha = std::modf(alpha, &index);
    int i0 = int(index) % int(values_.size());
    int i1 = (i0 + 1) % int(values_.size());

    auto &value0 = values_[i0];
    auto &value1 = values_[i1];

    return Mix(value0, value1, alpha);
  }

 private:
  std::vector<Ty> values_;
};
