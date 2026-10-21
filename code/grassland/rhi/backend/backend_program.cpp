#include "grassland/rhi/backend/backend_program.h"

#include <unordered_set>

#include "grassland/rhi/backend/backend_binding_group.h"

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
  if (desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (const auto &group : desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  if (!desc_.vertex_shader || !desc_.fragment_shader)
    return false;
  if (desc_.vertex_shader->GetDesc().stage != SHADER_STAGE_VERTEX)
    return false;
  if (desc_.fragment_shader->GetDesc().stage != SHADER_STAGE_FRAGMENT)
    return false;
  if (desc_.color_formats.size() > 8 || (desc_.color_formats.empty() && !IsDepthFormat(desc_.depth_stencil.format)))
    return false;
  for (auto format : desc_.color_formats)
    if (format != IMAGE_FORMAT_RGBA8_UNORM && format != IMAGE_FORMAT_BGRA8_UNORM)
      return false;
  if (!desc_.blend_states.empty() && desc_.blend_states.size() != desc_.color_formats.size())
    return false;
  for (const auto &blend : desc_.blend_states) {
    for (auto factor : {blend.src_color, blend.dst_color, blend.src_alpha, blend.dst_alpha})
      if (factor < BLEND_FACTOR_ZERO || factor > BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
        return false;
    for (auto op : {blend.color_op, blend.alpha_op})
      if (op < BLEND_OP_ADD || op > BLEND_OP_MAX)
        return false;
  }
  const auto &depth = desc_.depth_stencil;
  if (depth.format != IMAGE_FORMAT_UNKNOWN && !IsDepthFormat(depth.format))
    return false;
  if ((depth.depth_test || depth.depth_write || depth.stencil_test) && !IsDepthFormat(depth.format))
    return false;
  if (depth.stencil_test && !HasStencil(depth.format))
    return false;
  if (depth.depth_compare < COMPARE_NEVER || depth.depth_compare > COMPARE_ALWAYS)
    return false;
  for (auto face : {depth.front, depth.back}) {
    if (face.compare < COMPARE_NEVER || face.compare > COMPARE_ALWAYS)
      return false;
    for (auto op : {face.fail_op, face.depth_fail_op, face.pass_op})
      if (op < STENCIL_KEEP || op > STENCIL_DECREMENT_WRAP)
        return false;
  }
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
