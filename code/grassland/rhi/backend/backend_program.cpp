#include "grassland/rhi/backend/backend_program.h"

#include <unordered_set>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_program_state.h"

namespace grassland::rhi::backend {
void IBackendProgram::InitializeProgram(IBackendDevice *device, const ProgramDesc &desc) {
  StateProgram().device_ = device;
  StateProgram().desc_ = desc;
  StateProgram().vertex_shader_ = ObjectPtr<IShader>::Retain(desc.vertex_shader);
  StateProgram().fragment_shader_ = ObjectPtr<IShader>::Retain(desc.fragment_shader);
}

const ProgramDesc &IBackendProgram::CommonGetDesc() const {
  return StateProgram().desc_;
}

IBackendDevice *IBackendProgram::Owner() const {
  return StateProgram().device_;
}

bool IBackendProgram::ValidateLayout() const {
  if (StateProgram().desc_.topology < PRIMITIVE_TOPOLOGY_TRIANGLE_LIST ||
      StateProgram().desc_.topology > PRIMITIVE_TOPOLOGY_POINT_LIST ||
      StateProgram().desc_.rasterizer.cull_mode < CULL_MODE_NONE ||
      StateProgram().desc_.rasterizer.cull_mode > CULL_MODE_BACK ||
      StateProgram().desc_.rasterizer.front_face < FRONT_FACE_COUNTER_CLOCKWISE ||
      StateProgram().desc_.rasterizer.front_face > FRONT_FACE_CLOCKWISE ||
      StateProgram().desc_.rasterizer.fill_mode < FILL_MODE_SOLID ||
      StateProgram().desc_.rasterizer.fill_mode > FILL_MODE_WIREFRAME)
    return false;
  if (StateProgram().desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (const auto &group : StateProgram().desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  if (!StateProgram().desc_.vertex_shader || !StateProgram().desc_.fragment_shader)
    return false;
  if (StateProgram().desc_.vertex_shader->GetStage() != SHADER_STAGE_VERTEX)
    return false;
  if (StateProgram().desc_.fragment_shader->GetStage() != SHADER_STAGE_FRAGMENT)
    return false;
  if (StateProgram().desc_.color_formats.size() > 8 ||
      (StateProgram().desc_.color_formats.empty() && !IsDepthFormat(StateProgram().desc_.depth_stencil.format)))
    return false;
  for (auto format : StateProgram().desc_.color_formats)
    if (!IsColorFormat(format))
      return false;
  if (!StateProgram().desc_.blend_states.empty() &&
      StateProgram().desc_.blend_states.size() != StateProgram().desc_.color_formats.size())
    return false;
  for (size_t i = 0; i < StateProgram().desc_.blend_states.size(); ++i) {
    if (StateProgram().desc_.blend_states[i].blend_enable &&
        (IsSignedIntegerFormat(StateProgram().desc_.color_formats[i]) ||
         IsUnsignedIntegerFormat(StateProgram().desc_.color_formats[i])))
      return false;
  }
  for (const auto &blend : StateProgram().desc_.blend_states) {
    for (auto factor : {blend.src_color, blend.dst_color, blend.src_alpha, blend.dst_alpha})
      if (factor < BLEND_FACTOR_ZERO || factor > BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
        return false;
    for (auto op : {blend.color_op, blend.alpha_op})
      if (op < BLEND_OP_ADD || op > BLEND_OP_MAX)
        return false;
  }
  const auto &depth = StateProgram().desc_.depth_stencil;
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
  for (const auto &attribute : StateProgram().desc_.vertex_attributes) {
    if (attribute.binding >= StateProgram().desc_.vertex_bindings.size())
      return false;
    auto size = InputTypeSize(attribute.type);
    auto stride = StateProgram().desc_.vertex_bindings[attribute.binding].stride;
    if (!size || attribute.offset > stride || size > stride - attribute.offset)
      return false;
    if (!locations.insert(attribute.location).second)
      return false;
  }
  return true;
}
}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendProgram, "4a916e47-6d7c-4f6b-93bc-ed51bba0eaec")

namespace grassland::rhi::backend {
ProgramState::ProgramState() = default;
ProgramState::~ProgramState() = default;
}  // namespace grassland::rhi::backend
