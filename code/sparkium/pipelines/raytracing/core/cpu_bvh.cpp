#include "sparkium/pipelines/raytracing/core/cpu_bvh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace sparkium::raytracing {
void CpuBounds::Extend(const CpuBounds &b) {
  lo = glm::min(lo, b.lo);
  hi = glm::max(hi, b.hi);
}

void CpuBounds::Extend(glm::vec3 p) {
  lo = glm::min(lo, p);
  hi = glm::max(hi, p);
}

namespace {
constexpr unsigned bins_count = 12;

float Area(const CpuBounds &b) {
  glm::vec3 d = glm::max(b.hi - b.lo, glm::vec3(0));
  return d.x * d.y + d.y * d.z + d.z * d.x;
}

template <class T>
T Read(const std::vector<uint8_t> &bytes, size_t offset) {
  if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
    throw std::out_of_range("CPU BVH geometry range");
  T value;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}
}  // namespace

CpuBvhTree BuildCpuBvh(const std::vector<CpuBounds> &bounds, uint32_t leaf_size) {
  if (!leaf_size)
    throw std::invalid_argument("CPU BVH leaf size must be positive");
  if (bounds.size() > (std::numeric_limits<uint32_t>::max() - 16ull) / 68)
    throw std::overflow_error("CPU BVH exceeds 32-bit byte offsets");
  std::vector<uint32_t> ids(bounds.size());
  std::iota(ids.begin(), ids.end(), 0u);
  std::vector<CpuBvhNode> nodes(1);
  std::function<void(uint32_t, uint32_t, uint32_t, unsigned)> build = [&](uint32_t node, uint32_t begin, uint32_t end,
                                                                          unsigned depth) {
    CpuBounds box, centers;
    for (uint32_t i = begin; i < end; ++i) {
      box.Extend(bounds[ids[i]]);
      centers.Extend((bounds[ids[i]].lo + bounds[ids[i]].hi) * 0.5f);
    }
    nodes[node] = {box.lo, begin, box.hi, end - begin};
    if (end - begin <= leaf_size || depth >= 48)
      return;
    float best = std::numeric_limits<float>::infinity();
    int axis = -1;
    unsigned split = 0;
    for (int a = 0; a < 3; ++a) {
      float extent = centers.hi[a] - centers.lo[a];
      if (!(extent > 1e-20f))
        continue;
      std::array<CpuBounds, bins_count> bins;
      std::array<uint32_t, bins_count> counts{};
      auto bin = [&](uint32_t id) {
        float c = (bounds[id].lo[a] + bounds[id].hi[a]) * 0.5f;
        return std::min(bins_count - 1, unsigned((c - centers.lo[a]) / extent * bins_count));
      };
      for (uint32_t i = begin; i < end; ++i) {
        unsigned b = bin(ids[i]);
        bins[b].Extend(bounds[ids[i]]);
        ++counts[b];
      }
      std::array<float, bins_count> left_cost{};
      CpuBounds left;
      uint32_t left_count = 0;
      for (unsigned b = 0; b < bins_count; ++b) {
        left.Extend(bins[b]);
        left_count += counts[b];
        left_cost[b] = Area(left) * left_count;
      }
      CpuBounds right;
      uint32_t right_count = 0;
      for (unsigned b = bins_count - 1; b > 0; --b) {
        right.Extend(bins[b]);
        right_count += counts[b];
        float cost = left_cost[b - 1] + Area(right) * right_count;
        if (cost < best) {
          best = cost;
          axis = a;
          split = b;
        }
      }
    }
    uint32_t mid = begin;
    if (axis >= 0) {
      float extent = centers.hi[axis] - centers.lo[axis];
      auto it = std::stable_partition(ids.begin() + begin, ids.begin() + end, [&](uint32_t id) {
        float c = (bounds[id].lo[axis] + bounds[id].hi[axis]) * 0.5f;
        return std::min(bins_count - 1, unsigned((c - centers.lo[axis]) / extent * bins_count)) < split;
      });
      mid = uint32_t(it - ids.begin());
    }
    if (mid == begin || mid == end)
      mid = begin + (end - begin) / 2;
    uint32_t child = uint32_t(nodes.size());
    nodes.resize(nodes.size() + 2);
    nodes[node].first = child;
    nodes[node].count = 0;
    build(child, begin, mid, depth + 1);
    build(child + 1, mid, end, depth + 1);
  };

  if (!ids.empty())
    build(0, 0, uint32_t(ids.size()), 0);
  else
    nodes[0] = {glm::vec3(1), 0, glm::vec3(-1), 1};
  CpuBvhTree result;
  if (!ids.empty()) {
    result.bounds.lo = nodes[0].lo;
    result.bounds.hi = nodes[0].hi;
  }

  uint32_t index_offset = uint32_t(16 + nodes.size() * sizeof(CpuBvhNode));
  std::array<uint32_t, 4> header{uint32_t(nodes.size()), index_offset, uint32_t(ids.size()), 0};
  result.bytes.resize(index_offset + ids.size() * sizeof(uint32_t));
  std::memcpy(result.bytes.data(), header.data(), 16);
  std::memcpy(result.bytes.data() + 16, nodes.data(), nodes.size() * sizeof(CpuBvhNode));
  if (!ids.empty())
    std::memcpy(result.bytes.data() + index_offset, ids.data(), ids.size() * sizeof(uint32_t));
  return result;
}

CpuBvhTree BuildCpuMeshBvh(const std::vector<uint8_t> &geometry, uint32_t count) {
  uint32_t position = Read<uint32_t>(geometry, 8), stride = Read<uint32_t>(geometry, 12),
           indices = Read<uint32_t>(geometry, 48);
  std::vector<CpuBounds> bounds(count);
  for (uint32_t i = 0; i < count; ++i)
    for (uint32_t v = 0; v < 3; ++v) {
      uint32_t index = Read<uint32_t>(geometry, size_t(indices) + (size_t(i) * 3 + v) * 4);
      auto p = Read<glm::vec3>(geometry, size_t(position) + size_t(stride) * index);
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        throw std::runtime_error("CPU BVH requires finite positions");
      bounds[i].Extend(p);
    }
  return BuildCpuBvh(bounds);
}
}  // namespace sparkium::raytracing
