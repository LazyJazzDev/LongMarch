#include "grassland/rhi/rhix/rhix_ray_tracing_program.h"

namespace grassland::rhix {
RayTracingProgram::RayTracingProgram() noexcept = default;

RayTracingProgram::RayTracingProgram(rhi::IRayTracingProgram *object) noexcept : ObjectContainer(object) {
}

rhi::IRayTracingProgram *RayTracingProgram::Get() const noexcept {
  return static_cast<rhi::IRayTracingProgram *>(GetObject());
}

const rhi::RayTracingProgramDesc &RayTracingProgram::GetDesc() const {
  return Get()->GetDesc();
}

bool RayTracingProgram::Finalize() const {
  return Get()->Finalize();
}

bool RayTracingProgram::IsFinalized() const {
  return Get()->IsFinalized();
}

ShaderBindingTable RayTracingProgram::CreateShaderBindingTable(const rhi::ShaderBindingTableDesc &desc) const {
  return Get()->CreateShaderBindingTable(desc);
}

}  // namespace grassland::rhix
