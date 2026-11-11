#pragma once
#include "grassland/rhi/rhi_binding_group.h"
#include "grassland/rhi/rhi_shader.h"

LONGMARCH_RHI_UUID(grassland::rhi::IComputeProgram, "c1726e27-b6b5-4af9-b022-b27a814389cf")

namespace grassland::rhi {
struct ComputeProgramDesc {
  IShader *compute_shader{};
  PipelineLayoutDesc pipeline_layout;
};

// Creation retains the shader and returns a ready-to-bind pipeline, or nullptr.
class IComputeProgram : public IObject {
 public:
  virtual const ComputeProgramDesc &GetDesc() const = 0;

 protected:
  ~IComputeProgram() override;
};
}  // namespace grassland::rhi
