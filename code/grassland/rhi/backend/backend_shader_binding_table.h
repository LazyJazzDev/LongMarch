#pragma once
#include <array>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_buffer.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::IBackendShaderBindingTable,
                        grassland::rhi::IObject,
                        "bf62c599-58f1-4e20-b3b8-b9b23eac18bc")

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

namespace grassland::rhi::backend {
class ShaderBindingTableImpl final : public Object<rhi::IShaderBindingTable, IBackendShaderBindingTable> {
 public:
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
