#pragma once
#include "grassland/rhi/rhi_ray_tracing_program.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class ShaderBindingTable : public ObjectContainer {
 public:
  ShaderBindingTable() noexcept = default;

  ShaderBindingTable(rhi::IShaderBindingTable *object) noexcept : ObjectContainer(object) {
  }

  rhi::IShaderBindingTable *Get() const noexcept {
    return static_cast<rhi::IShaderBindingTable *>(GetObject());
  }

  const rhi::ShaderBindingTableDesc &GetDesc() const {
    return Get()->GetDesc();
  }
};

class RayTracingProgram : public ObjectContainer {
 public:
  RayTracingProgram() noexcept;
  RayTracingProgram(rhi::IRayTracingProgram *object) noexcept;
  rhi::IRayTracingProgram *Get() const noexcept;
  ShaderBindingTable CreateShaderBindingTable(const rhi::ShaderBindingTableDesc &desc) const;
  const rhi::RayTracingProgramDesc &GetDesc() const;
};
}  // namespace grassland::rhix
