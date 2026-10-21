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

bool ComputeProgram::IsFinalized() const {
  return finalized_;
}

Device *ComputeProgram::Owner() const {
  return device_;
}

bool ComputeProgram::ValidateLayout() const {
  if (!desc_.compute_shader || desc_.compute_shader->GetDesc().stage != SHADER_STAGE_COMPUTE ||
      desc_.pipeline_layout.groups.size() > 4)
    return false;
  for (const auto &group : desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  return true;
}
}  // namespace grassland::rhi::backend
