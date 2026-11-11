#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/metal/metal_program_base.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalProgram,
                        grassland::rhi::backend::metal::IMetalProgramBase,
                        "26dfc710-560c-4dff-bfbd-2adc53cd3739")

namespace grassland::rhi::backend::metal {
class IMetalProgram : public IMetalProgramBase {
 public:
  static rhi::IProgram *Create(IMetalDevice *device, const ProgramDesc &desc);
  virtual MTL::RenderPipelineState *NativePipeline() const = 0;
  virtual MTL::DepthStencilState *NativeDepthStencil() const = 0;
};

}  // namespace grassland::rhi::backend::metal
