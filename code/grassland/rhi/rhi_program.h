#pragma once
#include "grassland/rhi/rhi_binding_group.h"
#include "grassland/rhi/rhi_shader.h"

namespace grassland::rhi {
struct VertexBindingDesc {
  uint32_t stride{};
  bool per_instance{};
};

struct VertexAttributeDesc {
  uint32_t location{};
  uint32_t binding{};
  InputType format{INPUT_TYPE_FLOAT3};
  uint32_t offset{};
};

struct ProgramDesc {
  IShader *vertex_shader{};
  IShader *fragment_shader{};
  std::vector<VertexBindingDesc> vertex_bindings;
  std::vector<VertexAttributeDesc> vertex_attributes;
  // Empty means no color attachments.
  std::vector<ImageFormat> color_formats{IMAGE_FORMAT_R8G8B8A8_UNORM};
  // Empty disables blending. Otherwise provide one state per color attachment.
  std::vector<BlendState> blend_states;
  DepthStencilDesc depth_stencil;
  PipelineLayoutDesc pipeline_layout;
};

// Immutable and ready to bind when creation succeeds; failure returns nullptr.
// Current graphics programs use triangle-list topology and up to eight color attachments.
class IProgram : public IObject {
 public:
  virtual const ProgramDesc &GetDesc() const = 0;

 protected:
  ~IProgram() override;
};
}  // namespace grassland::rhi
