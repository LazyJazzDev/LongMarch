#pragma once
#include "grassland/rhi/backend/metal/metal_program_base.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
struct MetalProgramBaseState final {
  MetalProgramBaseState();
  ~MetalProgramBaseState();
  MTL::Device *native_device_{};
  // Borrows the immutable descriptor in the concrete Program base.
  const PipelineLayoutDesc *layout_{};
};
}  // namespace grassland::rhi::backend::metal
