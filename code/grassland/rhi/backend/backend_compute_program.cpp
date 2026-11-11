#include "grassland/rhi/backend/backend_compute_program.h"

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_compute_program_state.h"

namespace grassland::rhi::backend {
void IBackendComputeProgram::InitializeComputeProgram(IBackendDevice *device, const ComputeProgramDesc &desc) {
  StateComputeProgram().device_ = device;
  StateComputeProgram().desc_ = desc;
  StateComputeProgram().shader_ = ObjectPtr<IShader>::Retain(desc.compute_shader);
}

const ComputeProgramDesc &IBackendComputeProgram::CommonGetDesc() const {
  return StateComputeProgram().desc_;
}

IBackendDevice *IBackendComputeProgram::Owner() const {
  return StateComputeProgram().device_;
}

bool IBackendComputeProgram::ValidateLayout() const {
  if (!StateComputeProgram().desc_.compute_shader ||
      StateComputeProgram().desc_.compute_shader->GetStage() != SHADER_STAGE_COMPUTE ||
      StateComputeProgram().desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (const auto &group : StateComputeProgram().desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  return true;
}
}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendComputeProgram, "f64ab67b-e062-4368-8605-dc1700c88e59")

namespace grassland::rhi::backend {
ComputeProgramState::ComputeProgramState() = default;
ComputeProgramState::~ComputeProgramState() = default;
}  // namespace grassland::rhi::backend
