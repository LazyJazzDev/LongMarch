#pragma once

// Host reconstruction of the software BVH that `shaders/software/build.hlsl`
// builds on the GPU.
//
// The native backends traverse the tree with `native_traversal.h`, the very
// same code the GPU software path runs, so the tree itself has to match bit for
// bit: power-of-two leaf counts, the same leaf slot addressing, the same Morton
// keys, the same bitonic ordering and the same bottom-up reduction order. Every
// step below is a direct transcription of the corresponding compute kernel.

#include <cstdint>
#include <vector>

#include "sparkium/pipelines/native/shared/native_traversal.h"

namespace sparkium::native {

// One entry of the flattened instance table, matching the 112-byte GPU record.
struct BvhInstance {
  float3x4 object_to_world;
  float3x4 world_to_object;
  uint32_t root;
  uint32_t geometry;
  uint32_t material;
  uint32_t primitive_count;
};

// Node range reserved for one unique geometry, as assigned by
// `SoftwarePipeline::Update`.
struct BvhGeometry {
  const std::vector<uint32_t> *data;  // Flattened geometry blob.
  uint32_t root;
  uint32_t leaves;
  uint32_t count;
};

// Rounds up to a power of two, mirroring `SoftwarePipeline::LeafCount`.
uint32_t LeafCount(size_t count);

class BvhBuilder {
 public:
  // Builds every BLAS followed by the TLAS over `instances`.
  void Build(const std::vector<BvhGeometry> &geometries, const std::vector<BvhInstance> &instances,
             uint32_t tlas_leaves, uint32_t node_count);

  const std::vector<uint32_t> &Nodes() const {
    return nodes_;
  }

 private:
  void StoreNode(uint32_t index, const SoftwareNode &node);
  SoftwareNode LoadNode(uint32_t index) const;

  // `build.hlsl::WriteLeaf`, both the triangle and the instance variants.
  void WriteTriangleLeaf(const BvhGeometry &geometry, uint32_t slot, uint32_t primitive);
  void WriteInstanceLeaf(const std::vector<BvhInstance> &instances,
                         uint32_t root,
                         uint32_t leaves,
                         uint32_t primitive_count,
                         uint32_t slot,
                         uint32_t primitive);
  // `build.hlsl::ReduceNodes`, dispatched level by level.
  void Reduce(uint32_t root, uint32_t leaves);
  // `build.hlsl::{MortonKeys,BitonicSort}`; the result feeds `SortLeaves`.
  void SortKeys(uint32_t root, uint32_t leaves, uint32_t primitive_count);

  std::vector<uint32_t> nodes_;
  std::vector<uint2> keys_;
};

}  // namespace sparkium::native
