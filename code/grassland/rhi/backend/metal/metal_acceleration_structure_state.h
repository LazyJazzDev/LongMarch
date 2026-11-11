#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/metal/metal_acceleration_structure.h"

namespace grassland::rhi::backend::metal {
struct MetalAccelerationStructureState final {
  MetalAccelerationStructureState();
  ~MetalAccelerationStructureState();
  IMetalDevice *device_{};
  NS::SharedPtr<MTL::AccelerationStructure> structure_{};
  NS::SharedPtr<MTL::Buffer> query_{};
  uint64_t build_scratch_{}, update_scratch_{};
  bool allow_update_{}, allow_compaction_{}, compact_{};
};
}  // namespace grassland::rhi::backend::metal
