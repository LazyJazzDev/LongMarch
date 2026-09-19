#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

namespace sparkium::raytracing {
// CPU-only SAH tree. A tree starts with a 16-byte header, followed by 32-byte
// nodes and a packed primitive-index array. GPU construction/layout is independent.
struct CpuBounds {
  glm::vec3 lo{3.402823466e38f}, hi{-3.402823466e38f};
  void Extend(const CpuBounds &b);
  void Extend(glm::vec3 point);
};

struct CpuBvhNode {
  glm::vec3 lo;
  uint32_t first;
  glm::vec3 hi;
  uint32_t count;  // zero: children first and first+1; nonzero: index-array range
};

static_assert(sizeof(CpuBvhNode) == 32 && offsetof(CpuBvhNode, hi) == 16);

struct CpuBvhTree {
  CpuBounds bounds;
  std::vector<uint8_t> bytes;
};

CpuBvhTree BuildCpuBvh(const std::vector<CpuBounds> &bounds, uint32_t leaf_size = 4);
CpuBvhTree BuildCpuMeshBvh(const std::vector<uint8_t> &geometry, uint32_t primitive_count);
}  // namespace sparkium::raytracing
