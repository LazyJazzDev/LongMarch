#pragma once
#include "grassland/rhi/rhi_binding_group.h"
#include "grassland/rhi/rhi_shader.h"

namespace grassland::rhi {
struct ComputeProgramDesc {
  IShader *compute_shader{};
  PipelineLayoutDesc pipeline_layout;
};

// Immutable after successful Finalize; creation retains the shader.
class IComputeProgram : public IObject {
 public:
  virtual bool Finalize() = 0;
  virtual bool IsFinalized() const = 0;
  virtual const ComputeProgramDesc &GetDesc() const = 0;

 protected:
  ~IComputeProgram() override;
};
}  // namespace grassland::rhi
