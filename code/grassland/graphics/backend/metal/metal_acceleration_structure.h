#pragma once
#include "grassland/graphics/backend/metal/metal_util.h"

namespace grassland::graphics::backend {

// Native AS storage for inline ray queries. Pipeline RT/SBT dispatch remains unsupported.
class MetalAccelerationStructure : public AccelerationStructure {
 public:
  MetalAccelerationStructure(MetalCore *core,
                             BufferRange vertices,
                             BufferRange indices,
                             uint32_t vertex_count,
                             uint32_t stride,
                             uint32_t triangle_count,
                             RayTracingGeometryFlag flags);
  MetalAccelerationStructure(MetalCore *core,
                             BufferRange aabbs,
                             uint32_t stride,
                             uint32_t count,
                             RayTracingGeometryFlag flags);
  MetalAccelerationStructure(MetalCore *core, const std::vector<RayTracingInstance> &instances);
  int UpdateInstances(const std::vector<RayTracingInstance> &instances) override;

  MTL::AccelerationStructure *Handle() const {
    return structure_.get();
  }

  const std::vector<NS::SharedPtr<MTL::AccelerationStructure>> &Children() const {
    return children_;
  }

 private:
  void Build(MTL::AccelerationStructureDescriptor *descriptor);
  MetalCore *core_;
  bool top_level_ = false;
  NS::SharedPtr<MTL::AccelerationStructure> structure_;
  std::vector<NS::SharedPtr<MTL::AccelerationStructure>> children_;
  std::vector<MTL::AccelerationStructureUserIDInstanceDescriptor> instances_;
};

}  // namespace grassland::graphics::backend
