#pragma once
#include "grassland/rhi/backend/backend_shader_binding_table.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
class ShaderBindingTableImpl final : public Object<rhi::IShaderBindingTable, IBackendShaderBindingTable> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IShaderBindingTable, IBackendShaderBindingTable, IShaderBindingTable)
  ShaderBindingTableImpl(IRayTracingProgram *program,
                         const ShaderBindingTableDesc &desc,
                         ObjectPtr<IBuffer> buffer,
                         const std::array<Region, 4> &regions);

  const ShaderBindingTableDesc &GetDesc() const override {
    return desc_;
  }

  IRayTracingProgram *GetProgram() const override {
    return program_;
  }

  const std::array<Region, 4> &Regions() const {
    return regions_;
  }

 private:
  IRayTracingProgram *program_;
  ShaderBindingTableDesc desc_;
  ObjectPtr<IBuffer> buffer_;
  std::array<Region, 4> regions_;

 public:
  ~ShaderBindingTableImpl() override = default;
};
}  // namespace grassland::rhi::backend
