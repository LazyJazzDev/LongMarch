#pragma once
#include "grassland/rhi/rhi_ray_tracing_program.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class RayTracingProgram : public ObjectContainer {
 public:
  RayTracingProgram() noexcept;
  RayTracingProgram(rhi::IRayTracingProgram *object) noexcept;
  rhi::IRayTracingProgram *Get() const noexcept;
  bool Finalize() const;
  bool IsFinalized() const;
  const rhi::RayTracingProgramDesc &GetDesc() const;
};
}  // namespace grassland::rhix
