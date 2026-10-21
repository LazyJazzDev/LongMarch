#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_compute_program.h"

namespace grassland::rhi::backend {
class Device;

class ComputeProgram : public Object<IComputeProgram> {
 public:
  ComputeProgram(Device *device, const ComputeProgramDesc &desc);
  const ComputeProgramDesc &GetDesc() const override;
  bool IsFinalized() const override;
  Device *Owner() const;

 protected:
  bool ValidateLayout() const;
  Device *device_;
  ComputeProgramDesc desc_;
  bool finalized_{};

 private:
  ObjectPtr<IShader> shader_;
};
}  // namespace grassland::rhi::backend
