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

rhi::IShaderBindingTable *IBackendShaderBindingTable::Create(IRayTracingProgram *program,
                                                             const ShaderBindingTableDesc &desc,
                                                             ObjectPtr<IBuffer> buffer,
                                                             const std::array<Region, 4> &regions) {
  return new ShaderBindingTableImpl(program, desc, buffer, regions);
}
}  // namespace grassland::rhi::backend
