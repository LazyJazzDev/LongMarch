#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_program.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalProgram final : public Program {
 public:
  MetalProgram(MetalDevice *device, const ProgramDesc &desc);
  bool Finalize() override;
  MTL::RenderPipelineState *NativePipeline() const;

 private:
  NS::SharedPtr<MTL::RenderPipelineState> pipeline_;
};
}  // namespace grassland::rhi::backend::metal
