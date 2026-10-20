#include "grassland/rhi/backend/backend_program.h"

#include <unordered_set>

namespace grassland::rhi::backend {
Program::Program(Device *device, const ProgramDesc &desc)
    : device_(device),
      desc_(desc),
      vertex_shader_(ObjectPtr<IShader>::Retain(desc.vertex_shader)),
      fragment_shader_(ObjectPtr<IShader>::Retain(desc.fragment_shader)) {
}

Program::~Program() = default;

const ProgramDesc &Program::GetDesc() const {
  return desc_;
}

bool Program::IsFinalized() const {
  return finalized_;
}

Device *Program::Owner() const {
  return device_;
}

bool Program::ValidateLayout() const {
  if (!desc_.vertex_shader || !desc_.fragment_shader)
    return false;
  if (desc_.vertex_shader->GetDesc().stage != SHADER_STAGE_VERTEX)
    return false;
  if (desc_.fragment_shader->GetDesc().stage != SHADER_STAGE_FRAGMENT)
    return false;
  if (desc_.color_format != IMAGE_FORMAT_RGBA8_UNORM && desc_.color_format != IMAGE_FORMAT_BGRA8_UNORM)
    return false;
  std::unordered_set<uint32_t> locations;
  for (const auto &attribute : desc_.vertex_attributes) {
    if (attribute.binding >= desc_.vertex_bindings.size())
      return false;
    auto size = VertexFormatSize(attribute.format);
    auto stride = desc_.vertex_bindings[attribute.binding].stride;
    if (!size || attribute.offset > stride || size > stride - attribute.offset)
      return false;
    if (!locations.insert(attribute.location).second)
      return false;
  }
  return true;
}
}  // namespace grassland::rhi::backend
