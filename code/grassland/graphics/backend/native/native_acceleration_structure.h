#pragma once
#include "grassland/graphics/acceleration_structure.h"
#include "native_memory.h"
#include "optix_device.h"

namespace grassland::graphics::backend {

class OptixAccelerationStructure final : public AccelerationStructure {
 public:
  OptixAccelerationStructure(OptixDevice *,
                             BufferRange vertices,
                             BufferRange indices,
                             uint32_t vertex_count,
                             uint32_t stride,
                             uint32_t primitive_count,
                             RayTracingGeometryFlag flags);
  OptixAccelerationStructure(OptixDevice *, const std::vector<RayTracingInstance> &);
  int UpdateInstances(const std::vector<RayTracingInstance> &) override;

  OptixTraversableHandle Handle() const {
    return handle_;
  }

  OptixDevice *Device() const {
    return device_;
  }

  bool IsTopLevel() const {
    return top_level_;
  }

 private:
  void Build(const OptixBuildInput &);
  OptixDevice *device_;
  bool top_level_{};
  OptixTraversableHandle handle_{};
  std::unique_ptr<NativeMemory> output_, instances_;
  std::vector<OptixInstance> previous_instances_;
};

}  // namespace grassland::graphics::backend
