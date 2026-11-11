#pragma once
#include <Metal/Metal.hpp>
#include <array>

#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/rhi_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalProgramBase,
                        grassland::rhi::IObject,
                        "48d78e4c-57e0-4ad1-9dc1-254d12006fc6")

namespace grassland::rhi::backend::metal {
// Shared argument-buffer layout and shader translation; no IObject/refcount base.
class IMetalProgramBase : public IObject {
 public:
  virtual const PipelineLayoutDesc &BindingLayout() const = 0;
};

bool CompileMetal(const IShader *shader,
                  const PipelineLayoutDesc &layout,
                  const std::vector<InputAttributeDesc> &attributes,
                  std::string &source,
                  std::string &entry,
                  std::array<uint32_t, 3> *threads);
}  // namespace grassland::rhi::backend::metal
