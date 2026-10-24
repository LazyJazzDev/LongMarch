#include "grassland/rhi/backend/backend_ray_tracing_program.h"

#include "grassland/rhi/backend/backend_binding_group.h"

namespace grassland::rhi::backend {
RayTracingProgram::RayTracingProgram(Device *device, const RayTracingProgramDesc &desc) : device_(device), desc_(desc) {
  auto retain = [&](IShader *shader) {
    if (shader)
      shaders_.push_back(ObjectPtr<IShader>::Retain(shader));
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
  return desc_;
}

Device *RayTracingProgram::Owner() const {
  return device_;
}

bool RayTracingProgram::ValidateLayout() const {
  auto stage = [](IShader *shader, ShaderStage expected) { return shader && shader->GetStage() == expected; };
  if (!stage(desc_.ray_generation, SHADER_STAGE_RAY_GENERATION) || !desc_.max_recursion_depth ||
      desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (auto shader : desc_.miss_shaders)
    if (!stage(shader, SHADER_STAGE_MISS))
      return false;
  for (const auto &group : desc_.hit_groups) {
    if (!group.closest_hit && !group.any_hit && !group.intersection)
      return false;
    if (group.closest_hit && !stage(group.closest_hit, SHADER_STAGE_CLOSEST_HIT))
      return false;
    if (group.any_hit && !stage(group.any_hit, SHADER_STAGE_ANY_HIT))
      return false;
    if (group.intersection && !stage(group.intersection, SHADER_STAGE_INTERSECTION))
      return false;
  }
  for (auto shader : desc_.callable_shaders)
    if (!stage(shader, SHADER_STAGE_CALLABLE))
      return false;
  for (const auto &group : desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  return true;
}

bool RayTracingProgram::ValidateShaderBindingTable(const ShaderBindingTableDesc &desc) const {
  for (auto index : desc.miss_shader_indices)
    if (index >= desc_.miss_shaders.size())
      return false;
  for (auto index : desc.hit_group_indices)
    if (index >= desc_.hit_groups.size())
      return false;
  for (auto index : desc.callable_shader_indices)
    if (index >= desc_.callable_shaders.size())
      return false;
  return true;
}

ShaderBindingTableDesc RayTracingProgram::DefaultShaderBindingTableDesc() const {
  ShaderBindingTableDesc result;
  auto fill = [](auto &indices, size_t count) {
    for (size_t i = 0; i < count; ++i)
      indices.push_back(uint32_t(i));
  };
  fill(result.miss_shader_indices, desc_.miss_shaders.size());
  fill(result.hit_group_indices, desc_.hit_groups.size());
  fill(result.callable_shader_indices, desc_.callable_shaders.size());
  return result;
}

}  // namespace grassland::rhi::backend
