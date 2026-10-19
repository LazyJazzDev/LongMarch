#pragma once
#include "grassland/rhi/rhi_shader.h"

namespace grassland::rhi {
struct VertexBindingDesc {
  uint32_t stride{};
  bool per_instance{};
};

struct VertexAttributeDesc {
  uint32_t location{};
  uint32_t binding{};
  VertexFormat format{VERTEX_FORMAT_FLOAT3};
  uint32_t offset{};
};

struct ProgramDesc {
  IShader *vertex_shader{};
  IShader *fragment_shader{};
  std::vector<VertexBindingDesc> vertex_bindings;
  std::vector<VertexAttributeDesc> vertex_attributes;
  ImageFormat color_format{IMAGE_FORMAT_RGBA8_UNORM};
};

// Immutable after successful Finalize. Shaders/layout are captured at creation.
// Current graphics programs use triangle-list topology and one color attachment.
class IProgram : public IObject {
 public:
  virtual bool Finalize() = 0;
  virtual bool IsFinalized() const = 0;
  virtual const ProgramDesc &GetDesc() const = 0;

 protected:
  ~IProgram() override;
};
}  // namespace grassland::rhi
