#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class MetalProgramBase;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalProgramBase)
#include <Metal/Metal.hpp>
#include <array>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi::backend::metal {
// Shared argument-buffer layout and shader translation; no IObject/refcount base.
struct MetalProgramBaseState;

class MetalProgramBase : public IObject {
 protected:
  void InitializeMetalProgramBase(MTL::Device *device, const PipelineLayoutDesc &layout);

 public:
  const PipelineLayoutDesc &BindingLayout() const;

 protected:
  NS::SharedPtr<MTL::Function> CompileFunction(const IShader *shader,
                                               const std::vector<InputAttributeDesc> &attributes = {},
                                               std::array<uint32_t, 3> *threads = nullptr) const;
  virtual MetalProgramBaseState &StateMetalProgramBase() = 0;
  virtual const MetalProgramBaseState &StateMetalProgramBase() const = 0;
};
}  // namespace grassland::rhi::backend::metal
