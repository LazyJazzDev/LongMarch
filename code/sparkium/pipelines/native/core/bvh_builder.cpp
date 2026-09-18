#include "sparkium/pipelines/native/core/bvh_builder.h"

#include <algorithm>
#include <stdexcept>

namespace sparkium::native {

namespace {
// `build.hlsl::EmptyNode`.
SoftwareNode EmptyNode() {
  SoftwareNode node;
  node.lo = float3{3.402823e38f, 3.402823e38f, 3.402823e38f};
  node.hi = -node.lo;
  node.first = node.second = SOFTWARE_INVALID;
  return node;
}

// `build.hlsl::SpreadBits`.
uint32_t SpreadBits(uint32_t v) {
  v = (v | (v << 16)) & 0x030000ffu;
  v = (v | (v << 8)) & 0x0300f00fu;
  v = (v | (v << 4)) & 0x030c30c3u;
  return (v | (v << 2)) & 0x09249249u;
}

ByteBuffer MakeBuffer(const std::vector<uint32_t> &data) {
  ByteBuffer buffer;
  buffer.data = data.data();
  buffer.size = static_cast<uint32_t>(data.size() * sizeof(uint32_t));
  return buffer;
}
}  // namespace

uint32_t LeafCount(size_t count) {
  if (count > (uint32_t(1) << 21))
    throw std::runtime_error("software BVH exceeds 2097152 leaves per tree");
  uint32_t result = 1;
  while (result < count)
    result *= 2;
  return result;
}

void BvhBuilder::StoreNode(uint32_t index, const SoftwareNode &node) {
  uint32_t *target = nodes_.data() + (static_cast<size_t>(index) * SOFTWARE_NODE_BYTES) / 4;
  target[0] = asuint(node.lo.x);
  target[1] = asuint(node.lo.y);
  target[2] = asuint(node.lo.z);
  target[3] = node.first;
  target[4] = asuint(node.hi.x);
  target[5] = asuint(node.hi.y);
  target[6] = asuint(node.hi.z);
  target[7] = node.second;
}

SoftwareNode BvhBuilder::LoadNode(uint32_t index) const {
  return LoadSoftwareNode(MakeBuffer(nodes_), index);
}

void BvhBuilder::WriteTriangleLeaf(const BvhGeometry &geometry, uint32_t slot, uint32_t primitive) {
  SoftwareNode node = EmptyNode();
  if (primitive < geometry.count) {
    node.second = primitive;
    const ByteBuffer buffer = MakeBuffer(*geometry.data);
    const uint32_t position_offset = buffer.Load(8), position_stride = buffer.Load(12);
    const uint3 ids = buffer.Load3(buffer.Load(48) + primitive * 12);
    for (uint32_t v = 0; v < 3; ++v) {
      const float3 p = LoadFloat3(buffer, position_offset + position_stride * ids[v]);
      node.lo = glm::min(node.lo, p);
      node.hi = glm::max(node.hi, p);
    }
  }
  StoreNode(geometry.root + geometry.leaves - 1 + slot, node);
}

void BvhBuilder::WriteInstanceLeaf(const std::vector<BvhInstance> &instances,
                                   uint32_t root,
                                   uint32_t leaves,
                                   uint32_t primitive_count,
                                   uint32_t slot,
                                   uint32_t primitive) {
  SoftwareNode node = EmptyNode();
  if (primitive < primitive_count) {
    node.second = primitive;
    const BvhInstance &instance = instances[primitive];
    const SoftwareNode local = LoadNode(instance.root);
    if (local.lo.x <= local.hi.x && local.lo.y <= local.hi.y && local.lo.z <= local.hi.z) {
      for (uint32_t corner = 0; corner < 8; ++corner) {
        const float3 corner_position{(corner & 1) ? local.hi.x : local.lo.x, (corner & 2) ? local.hi.y : local.lo.y,
                                     (corner & 4) ? local.hi.z : local.lo.z};
        const float3 p = mul(instance.object_to_world, float4{corner_position, 1.0f});
        node.lo = glm::min(node.lo, p);
        node.hi = glm::max(node.hi, p);
      }
    }
  }
  StoreNode(root + leaves - 1 + slot, node);
}

void BvhBuilder::Reduce(uint32_t root, uint32_t leaves) {
  for (uint32_t count = leaves / 2; count; count /= 2) {
    const uint32_t level_first = count - 1;
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t index = level_first + i;
      SoftwareNode node;
      node.first = root + index * 2 + 1;
      node.second = node.first + 1;
      const SoftwareNode a = LoadNode(node.first), b = LoadNode(node.second);
      node.lo = glm::min(a.lo, b.lo);
      node.hi = glm::max(a.hi, b.hi);
      StoreNode(root + index, node);
    }
  }
}

void BvhBuilder::SortKeys(uint32_t root, uint32_t leaves, uint32_t primitive_count) {
  keys_.assign(leaves, uint2{SOFTWARE_INVALID, 0});
  const SoftwareNode bounds = LoadNode(root);
  for (uint32_t i = 0; i < leaves; ++i) {
    uint32_t code = SOFTWARE_INVALID;
    if (i < primitive_count) {
      const SoftwareNode node = LoadNode(root + leaves - 1 + i);
      const float3 center = (node.lo + node.hi) * 0.5f;
      const float3 extent = glm::max(bounds.hi - bounds.lo, make_float3(1.0e-20f));
      const float3 scaled = saturate((center - bounds.lo) / extent) * 1023.0f;
      const uint3 p{static_cast<uint32_t>(scaled.x), static_cast<uint32_t>(scaled.y), static_cast<uint32_t>(scaled.z)};
      code = SpreadBits(p.x) | (SpreadBits(p.y) << 1) | (SpreadBits(p.z) << 2);
    }
    keys_[i] = uint2{code, i};
  }
  // The GPU runs one dispatch per (stage, stride) pair with a barrier between
  // them, so the comparator sees the values written by the previous pass only.
  for (uint32_t stage = 2; stage <= leaves; stage *= 2) {
    for (uint32_t stride = stage / 2; stride; stride /= 2) {
      for (uint32_t i = 0; i < leaves; ++i) {
        const uint32_t j = i ^ stride;
        if (j <= i)
          continue;
        const uint2 a = keys_[i], b = keys_[j];
        const bool greater = a.x > b.x || (a.x == b.x && a.y > b.y);
        if (greater == ((i & stage) == 0)) {
          keys_[i] = b;
          keys_[j] = a;
        }
      }
    }
  }
}

void BvhBuilder::Build(const std::vector<BvhGeometry> &geometries,
                       const std::vector<BvhInstance> &instances,
                       uint32_t tlas_leaves,
                       uint32_t node_count) {
  nodes_.assign(static_cast<size_t>(node_count) * SOFTWARE_NODE_BYTES / 4, 0);

  // Bottom-level trees first: the instance leaves read their root boxes.
  for (const auto &geometry : geometries) {
    for (uint32_t slot = 0; slot < geometry.leaves; ++slot)
      WriteTriangleLeaf(geometry, slot, slot);
    Reduce(geometry.root, geometry.leaves);
    if (geometry.leaves > 1) {
      SortKeys(geometry.root, geometry.leaves, geometry.count);
      for (uint32_t slot = 0; slot < geometry.leaves; ++slot)
        WriteTriangleLeaf(geometry, slot, keys_[slot].y);
      Reduce(geometry.root, geometry.leaves);
    }
  }

  const uint32_t instance_count = static_cast<uint32_t>(instances.size());
  for (uint32_t slot = 0; slot < tlas_leaves; ++slot)
    WriteInstanceLeaf(instances, 0, tlas_leaves, instance_count, slot, slot);
  Reduce(0, tlas_leaves);
  if (tlas_leaves > 1) {
    SortKeys(0, tlas_leaves, instance_count);
    for (uint32_t slot = 0; slot < tlas_leaves; ++slot)
      WriteInstanceLeaf(instances, 0, tlas_leaves, instance_count, slot, keys_[slot].y);
    Reduce(0, tlas_leaves);
  }
}

}  // namespace sparkium::native
