#pragma once
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::IBackendProgram,
                        grassland::rhi::IObject,
                        "4a916e47-6d7c-4f6b-93bc-ed51bba0eaec")

namespace grassland::rhi::backend {
class IBackendProgram : public IObject {
 public:
  virtual IBackendDevice *Owner() const = 0;
};
}  // namespace grassland::rhi::backend
