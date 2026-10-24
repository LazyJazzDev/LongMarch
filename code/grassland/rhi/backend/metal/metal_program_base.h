#pragma once
#include <Metal/Metal.hpp>
#include <array>

#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi::backend::metal {
// Shared argument-buffer layout and shader translation; no IObject/refcount base.
class MetalProgramBase {
 public:
  MetalProgramBase(MTL::Device *device, const PipelineLayoutDesc &layout);
  const PipelineLayoutDesc &BindingLayout() const;

 protected:
  NS::SharedPtr<MTL::Function> CompileFunction(const IShader *shader,
                                               const std::vector<InputAttributeDesc> &attributes = {},
                                               std::array<uint32_t, 3> *threads = nullptr) const;

 private:
  MTL::Device *native_device_;
  // Borrows the immutable descriptor in the concrete Program base.
  const PipelineLayoutDesc &layout_;
};
}  // namespace grassland::rhi::backend::metal
