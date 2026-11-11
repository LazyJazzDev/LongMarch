#include "grassland/rhi/backend/backend_shader_binding_table.h"

#include "grassland/rhi/backend/backend_shader_binding_table_impl.h"

namespace grassland::rhi::backend {
ShaderBindingTableImpl::ShaderBindingTableImpl(IRayTracingProgram *program,
                                               const ShaderBindingTableDesc &desc,
                                               ObjectPtr<IBuffer> buffer,
                                               const std::array<Region, 4> &regions)
    : program_(program),
      desc_(desc),
      buffer_(std::move(buffer)),
      regions_(regions) {
}

IBackendShaderBindingTable *IBackendShaderBindingTable::Create(IRayTracingProgram *program,
                                                               const ShaderBindingTableDesc &desc,
                                                               ObjectPtr<IBuffer> buffer,
                                                               const std::array<Region, 4> &regions) {
  return new ShaderBindingTableImpl(program, desc, buffer, regions);
}
}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendShaderBindingTable, "bf62c599-58f1-4e20-b3b8-b9b23eac18bc")
