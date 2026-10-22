#pragma once
#include "grassland/rhi/rhi_binding_group.h"
#include "grassland/rhi/rhi_shader.h"

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
