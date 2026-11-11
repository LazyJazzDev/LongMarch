#include "grassland/rhi/backend/backend_ray_tracing_program.h"

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_ray_tracing_program_state.h"

namespace grassland::rhi::backend {
void RayTracingProgram::InitializeRayTracingProgram(Device *device, const RayTracingProgramDesc &desc) {
  StateRayTracingProgram().device_ = device;
  StateRayTracingProgram().desc_ = desc;

  auto retain = [&](IShader *shader) {
    if (shader)
      StateRayTracingProgram().shaders_.push_back(ObjectPtr<IShader>::Retain(shader));
  };
  retain(desc.ray_generation);
  for (auto shader : desc.miss_shaders)
    retain(shader);
  for (const auto &group : desc.hit_groups) {
    retain(group.closest_hit);
    retain(group.any_hit);
    retain(group.intersection);
  }
  for (auto shader : desc.callable_shaders)
    retain(shader);
}

const RayTracingProgramDesc &RayTracingProgram::GetDesc() const {
  return StateRayTracingProgram().desc_;
}

Device *RayTracingProgram::Owner() const {
  return StateRayTracingProgram().device_;
}

bool RayTracingProgram::ValidateLayout() const {
  auto stage = [](IShader *shader, ShaderStage expected) { return shader && shader->GetStage() == expected; };
  if (!stage(StateRayTracingProgram().desc_.ray_generation, SHADER_STAGE_RAY_GENERATION) ||
      !StateRayTracingProgram().desc_.max_recursion_depth ||
      StateRayTracingProgram().desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (auto shader : StateRayTracingProgram().desc_.miss_shaders)
    if (!stage(shader, SHADER_STAGE_MISS))
      return false;
  for (const auto &group : StateRayTracingProgram().desc_.hit_groups) {
    if (!group.closest_hit && !group.any_hit && !group.intersection)
      return false;
    if (group.closest_hit && !stage(group.closest_hit, SHADER_STAGE_CLOSEST_HIT))
      return false;
    if (group.any_hit && !stage(group.any_hit, SHADER_STAGE_ANY_HIT))
      return false;
    if (group.intersection && !stage(group.intersection, SHADER_STAGE_INTERSECTION))
      return false;
  }
  for (auto shader : StateRayTracingProgram().desc_.callable_shaders)
    if (!stage(shader, SHADER_STAGE_CALLABLE))
      return false;
  for (const auto &group : StateRayTracingProgram().desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  return true;
}

bool RayTracingProgram::ValidateShaderBindingTable(const ShaderBindingTableDesc &desc) const {
  for (auto index : desc.miss_shader_indices)
    if (index >= StateRayTracingProgram().desc_.miss_shaders.size())
      return false;
  for (auto index : desc.hit_group_indices)
    if (index >= StateRayTracingProgram().desc_.hit_groups.size())
      return false;
  for (auto index : desc.callable_shader_indices)
    if (index >= StateRayTracingProgram().desc_.callable_shaders.size())
      return false;
  return true;
}

ShaderBindingTableDesc RayTracingProgram::DefaultShaderBindingTableDesc() const {
  ShaderBindingTableDesc result;
  auto fill = [](auto &indices, size_t count) {
    for (size_t i = 0; i < count; ++i)
      indices.push_back(uint32_t(i));
  };
  fill(result.miss_shader_indices, StateRayTracingProgram().desc_.miss_shaders.size());
  fill(result.hit_group_indices, StateRayTracingProgram().desc_.hit_groups.size());
  fill(result.callable_shader_indices, StateRayTracingProgram().desc_.callable_shaders.size());
  return result;
}

}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::RayTracingProgram, "e0b9733f-8a37-4702-800c-3022865d522c")

namespace grassland::rhi::backend {
RayTracingProgramState::RayTracingProgramState() = default;
RayTracingProgramState::~RayTracingProgramState() = default;
}  // namespace grassland::rhi::backend
