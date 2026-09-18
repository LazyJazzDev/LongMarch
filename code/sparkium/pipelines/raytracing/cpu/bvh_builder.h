// Host-side BVH construction for the CPU backend.
//
// The GPU fallback builds its trees with compute passes that reduce, Morton
// code and bitonic sort the leaves, which is the fastest option when the sort
// itself runs in parallel. On a CPU nothing of that is worth doing: a binned
// SAH sweep is both cheaper to build and better quality than a Morton order,
// and quality matters more here because the traversal is the whole cost.
//
// The node layout is not ours to change though. software/traversal.hlsli walks
// 32-byte nodes with `first`/`second` child indices and marks leaves by
// first == SOFTWARE_INVALID, with `second` holding the primitive index, so any
// binary tree over that layout is traversable. This builder emits exactly that,
// one primitive per leaf.
#pragma once

#include <cstdint>
#include <vector>

namespace sparkium::raytracing::cpu {

inline constexpr uint32_t kSoftwareInvalid = 0xffffffffu;

// Mirrors SoftwareNode in software/layout.hlsli, which is 32 bytes.
struct SoftwareNode {
  float lo[3];
  uint32_t first;
  float hi[3];
  uint32_t second;
};
static_assert(sizeof(SoftwareNode) == 32, "HLSL software node layout changed");

// A primitive bound by the tree. `index` is what the leaf reports: a triangle
// index inside a mesh tree, or an instance index inside the top-level tree.
struct BvhPrimitive {
  float lo[3];
  float hi[3];
  uint32_t index;
};

// Number of nodes a tree over `primitive_count` primitives occupies. A full
// binary tree over n leaves has 2n-1 nodes, and that is the whole tree here.
inline size_t BvhNodeCount(size_t primitive_count) {
  return primitive_count == 0 ? 0 : primitive_count * 2 - 1;
}

// Builds one tree over `primitives` into `nodes[root ...]`, writing
// BvhNodeCount(primitives.size()) nodes. `primitives` is reordered in place.
//
// Traversal uses a fixed 32-entry stack on the GPU, so the builder caps its
// depth and falls back to a median split rather than letting a degenerate SAH
// split produce a pathologically deep tree.
void BuildBvh(std::vector<SoftwareNode> &nodes, uint32_t root, std::vector<BvhPrimitive> &primitives);

}  // namespace sparkium::raytracing::cpu
