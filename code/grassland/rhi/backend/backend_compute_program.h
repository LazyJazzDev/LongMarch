#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class ComputeProgram;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::ComputeProgram)
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_compute_program.h"

namespace grassland::rhi::backend {
class Device;

class ComputeProgram : public Object<IComputeProgram> {
 public:
  LONGMARCH_RHI_QUERY(ComputeProgram, Object<IComputeProgram>)

  ComputeProgram(Device *device, const ComputeProgramDesc &desc);
  const ComputeProgramDesc &GetDesc() const override;
  Device *Owner() const;

 protected:
  bool ValidateLayout() const;
  Device *device_;
  ComputeProgramDesc desc_;

 private:
  ObjectPtr<IShader> shader_;
};
}  // namespace grassland::rhi::backend
