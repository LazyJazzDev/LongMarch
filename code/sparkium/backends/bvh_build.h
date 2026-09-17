#pragma once
// Host-side builder for the offline (CPU/CUDA) backends' bounding volume
// hierarchies.
//
// It reproduces the tree that the online software pipeline builds with
// code/sparkium/shaders/software/build.hlsl (see
// raytracing/SoftwarePipeline::AppendBuild): a perfectly balanced heap over
// `leaf_count` (the next power of two >= primitive count) Morton-sorted leaves,
// stored in the byte-compatible `SoftwareNode` layout with
//   node[root + i]  -> children node[root + 2i + 1], node[root + 2i + 2]
//   node[root + leaf_count - 1 + slot] -> leaf holding primitive `slot`
// Empty leaves keep the inverted bounds of build.hlsl's EmptyNode().
//
// Matching the online traversal order matters for reproducibility: the CPU,
// CUDA and Vulkan `rt_fallback` runs use the same BVH shape, so they disagree
// only through floating point rounding, not through different hit selection.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "sparkium/backends/core/structs.h"
#include "sparkium/backends/core/hlsl_math.h"

namespace sparkium::backends {

// Same limit as raytracing::LeafCount.
inline uint32_t LeafCountFor(uint32_t count) {
  if (count > (uint32_t(1) << 21))
    throw std::runtime_error("offline BVH exceeds 2097152 leaves per tree");
  uint32_t result = 1;
  while (result < count)
    result *= 2;
  return result;
}

namespace bvh_detail {

inline float3 LoadPosition(const uint8_t *mesh_data,
                           uint32_t position_offset,
                           uint32_t position_stride,
                           uint32_t index_offset,
                           uint32_t primitive,
                           uint32_t corner) {
  const uint32_t *words = reinterpret_cast<const uint32_t *>(mesh_data);
  uint32_t vertex = words[(index_offset + primitive * 12) / 4 + corner];
  uint32_t offset = (position_offset + position_stride * vertex) / 4;
  return float3(device::asfloat(words[offset]), device::asfloat(words[offset + 1]),
                device::asfloat(words[offset + 2]));
}

inline uint32_t SpreadBits(uint32_t v) {
  v = (v | (v << 16)) & 0x030000ff;
  v = (v | (v << 8)) & 0x0300f00f;
  v = (v | (v << 4)) & 0x030c30c3;
  return (v | (v << 2)) & 0x09249249;
}

inline uint32_t ReadWord(const uint8_t *data, uint32_t byte_offset) {
  uint32_t value;
  std::memcpy(&value, data + byte_offset, sizeof(value));
  return value;
}

inline float ReadFloat(const uint8_t *data, uint32_t byte_offset) {
  float value;
  std::memcpy(&value, data + byte_offset, sizeof(value));
  return value;
}

struct Leaf {
  float3 lo;
  float3 hi;
  uint32_t code{0};
  uint32_t index{0};
  bool valid{false};
};

inline SoftwareNode EmptyNode() {
  SoftwareNode node;
  node.lo = float3(3.402823e38f);
  node.hi = -node.lo;
  node.first = SPARKIUM_SOFTWARE_INVALID;
  node.second = SPARKIUM_SOFTWARE_INVALID;
  return node;
}

// Shared tail of both tree builders: Morton sort the (already computed) leaf
// boxes and emit leaves plus the balanced internal heap.
inline uint32_t EmitTree(std::vector<SoftwareNode> &nodes, std::vector<Leaf> &leaves) {
  const uint32_t leaf_count = LeafCountFor(static_cast<uint32_t>(leaves.size()));
  float3 root_lo(3.402823e38f), root_hi(-3.402823e38f);
  for (const Leaf &leaf : leaves) {
    if (!leaf.valid)
      continue;
    root_lo = device::min(root_lo, leaf.lo);
    root_hi = device::max(root_hi, leaf.hi);
  }
  float3 extent = device::max(root_hi - root_lo, float3(1.0e-20f));
  for (Leaf &leaf : leaves) {
    float3 center = (leaf.lo + leaf.hi) * 0.5f;
    float3 t = device::saturate((center - root_lo) / extent);
    uint32_t x = static_cast<uint32_t>(t.x * 1023.0f);
    uint32_t y = static_cast<uint32_t>(t.y * 1023.0f);
    uint32_t z = static_cast<uint32_t>(t.z * 1023.0f);
    leaf.code = SpreadBits(x) | (SpreadBits(y) << 1) | (SpreadBits(z) << 2);
  }
  std::vector<uint32_t> order(leaves.size());
  for (uint32_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    if (leaves[a].code != leaves[b].code)
      return leaves[a].code < leaves[b].code;
    return a < b;
  });

  const uint32_t root = static_cast<uint32_t>(nodes.size());
  nodes.resize(root + static_cast<size_t>(leaf_count) * 2 - 1);
  for (uint32_t slot = 0; slot < leaf_count; ++slot) {
    SoftwareNode node = EmptyNode();
    if (slot < order.size()) {
      const Leaf &leaf = leaves[order[slot]];
      node.second = leaf.index;
      node.lo = leaf.lo;
      node.hi = leaf.hi;
    }
    nodes[root + leaf_count - 1 + slot] = node;
  }
  for (uint32_t i = leaf_count - 1; i-- > 0;) {
    const SoftwareNode &a = nodes[root + i * 2 + 1];
    const SoftwareNode &b = nodes[root + i * 2 + 2];
    SoftwareNode node;
    node.lo = device::min(a.lo, b.lo);
    node.hi = device::max(a.hi, b.hi);
    node.first = root + i * 2 + 1;
    node.second = root + i * 2 + 2;
    nodes[root + i] = node;
  }
  return root;
}

}  // namespace bvh_detail

// Builds one tree for a mesh blob; returns the root node index.
inline uint32_t BuildMeshTree(std::vector<SoftwareNode> &nodes, const uint8_t *mesh_data, uint32_t primitive_count) {
  using namespace bvh_detail;
  const uint32_t position_offset = ReadWord(mesh_data, 8);
  const uint32_t position_stride = ReadWord(mesh_data, 12);
  const uint32_t index_offset = ReadWord(mesh_data, 48);
  std::vector<Leaf> leaves(primitive_count);
  for (uint32_t primitive = 0; primitive < primitive_count; ++primitive) {
    float3 lo(3.402823e38f), hi(-3.402823e38f);
    for (uint32_t corner = 0; corner < 3; ++corner) {
      float3 position = LoadPosition(mesh_data, position_offset, position_stride, index_offset, primitive, corner);
      lo = device::min(lo, position);
      hi = device::max(hi, position);
    }
    leaves[primitive] = {lo, hi, 0, primitive, true};
  }
  return EmitTree(nodes, leaves);
}

// Builds the top-level tree over instances into `nodes`. Leaf boxes bound the
// instance's mesh, mirroring the `instance_tree` branch of WriteLeaf, which
// reads the mesh tree node that was written into the same node buffer before
// the instance pass. The offline backends keep the two trees in separate
// arrays, so the mesh tree is passed explicitly.
inline uint32_t BuildInstanceTree(std::vector<SoftwareNode> &nodes,
                                 const SoftwareNode *mesh_nodes,
                                 const InstanceData *instances,
                                 uint32_t num_instances,
                                 const MeshRange *meshes) {
  using namespace bvh_detail;
  std::vector<Leaf> leaves(num_instances);
  for (uint32_t i = 0; i < num_instances; ++i) {
    const InstanceData &instance = instances[i];
    const MeshRange &mesh = meshes[instance.mesh];
    // Bounds of the mesh's local root node in object space.
    float3 local_lo = mesh_nodes[mesh.root].lo;
    float3 local_hi = mesh_nodes[mesh.root].hi;
    float3 lo(3.402823e38f), hi(-3.402823e38f);
    if (!(local_lo.x > local_hi.x || local_lo.y > local_hi.y || local_lo.z > local_hi.z)) {
      for (uint32_t corner = 0; corner < 8; ++corner) {
        float3 point((corner & 1) ? local_hi.x : local_lo.x, (corner & 2) ? local_hi.y : local_lo.y,
                     (corner & 4) ? local_hi.z : local_lo.z);
        float3 world = transform_point(instance.object_to_world, point);
        lo = device::min(lo, world);
        hi = device::max(hi, world);
      }
    }
    leaves[i] = {lo, hi, 0, i, true};
  }
  return EmitTree(nodes, leaves);
}

}  // namespace sparkium::backends
