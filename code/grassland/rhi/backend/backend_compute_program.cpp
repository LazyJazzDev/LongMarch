#include "grassland/rhi/backend/backend_compute_program.h"

#include "grassland/rhi/backend/backend_binding_group.h"

namespace grassland::rhi::backend {
ComputeProgram::ComputeProgram(Device *device, const ComputeProgramDesc &desc)
    : device_(device),
      desc_(desc),
      shader_(ObjectPtr<IShader>::Retain(desc.compute_shader)) {
}

const ComputeProgramDesc &ComputeProgram::GetDesc() const {
  return desc_;
}

Device *ComputeProgram::Owner() const {
  return device_;
}

bool ComputeProgram::ValidateLayout() const {
  if (!desc_.compute_shader || desc_.compute_shader->GetStage() != SHADER_STAGE_COMPUTE ||
      desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (const auto &group : desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  return true;
}
}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::ComputeProgram, "f64ab67b-e062-4368-8605-dc1700c88e59")
