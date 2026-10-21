#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/metal/metal_program_base.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalProgram final : public Program, public MetalProgramBase {
 public:
  MetalProgram(MetalDevice *device, const ProgramDesc &desc);
  bool Finalize() override;
  MTL::RenderPipelineState *NativePipeline() const;
  MTL::DepthStencilState *NativeDepthStencil() const;
  static constexpr uint32_t VertexBufferBase = 4;

 private:
  NS::SharedPtr<MTL::RenderPipelineState> pipeline_;
  NS::SharedPtr<MTL::DepthStencilState> depth_stencil_;
};
}  // namespace grassland::rhi::backend::metal
