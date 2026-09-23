#pragma once
#include <glm/glm.hpp>

#include "grassland/grassland.h"
#define LAND_PI 3.14159265358979323846
#define LOG_INFO grassland::LogInfo
#define LOG_ERROR grassland::LogError
#define LOG_WARN grassland::LogWarning

namespace geometry {
constexpr float eps = 1e-8;

template <class Ty>
bool IsZero(const Ty &val) {
  return glm::length(val) < eps;
}

template <class Ty>
bool Equal(const Ty &v0, const Ty &v1) {
  return IsZero(v0 - v1);
}

bool Between(float x0, float x1, float y);
}  // namespace geometry
