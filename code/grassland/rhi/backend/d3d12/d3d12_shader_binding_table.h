#pragma once
#include "grassland/rhi/backend/backend_shader_binding_table.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12ShaderBindingTable final : public ShaderBindingTable {
 public:
  void AddRef() noexcept override {
    RetainReference();
  }

  void Release() noexcept override {
    if (ReleaseReference())
      delete this;
  }

  const ShaderBindingTableDesc &GetDesc() const override {
    return TableDescription();
  }

  IRayTracingProgram *GetProgram() const override {
    return SourceProgram();
  }

  using ShaderBindingTable::ShaderBindingTable;
};
}  // namespace grassland::rhi::backend::d3d12
