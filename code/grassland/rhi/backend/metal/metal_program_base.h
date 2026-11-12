#pragma once
#include <Metal/Metal.hpp>
#include <array>

#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/rhi_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalProgramBase,
                        grassland::rhi::IObject,
                        "48d78e4c-57e0-4ad1-9dc1-254d12006fc6")

namespace grassland::rhi::backend::metal {
// Native program capabilities shared by graphics and compute command recording.
class IMetalProgramBase : public IObject {
 public:
  virtual IMetalDevice *Owner() const = 0;
  virtual const PipelineLayoutDesc &BindingLayout() const = 0;
};

NS::SharedPtr<MTL::Function> CompileMetalFunction(MTL::Device *device,
                                                  const PipelineLayoutDesc &layout,
                                                  const IShader *shader,
                                                  const std::vector<InputAttributeDesc> &attributes = {},
                                                  std::array<uint32_t, 3> *threads = nullptr);

}  // namespace grassland::rhi::backend::metal
