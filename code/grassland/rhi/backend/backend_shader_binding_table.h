#pragma once
#include <array>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_buffer.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendShaderBindingTable)

namespace grassland::rhi::backend {
class IBackendShaderBindingTable : public IObject {
 public:
  struct Region {
    uint64_t address{}, size{}, stride{};
  };

  static rhi::IShaderBindingTable *Create(IRayTracingProgram *program,
                                          const ShaderBindingTableDesc &desc,
                                          ObjectPtr<IBuffer> buffer,
                                          const std::array<Region, 4> &regions);
  virtual const std::array<Region, 4> &Regions() const = 0;
};

}  // namespace grassland::rhi::backend
