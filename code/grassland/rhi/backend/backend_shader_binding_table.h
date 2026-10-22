#pragma once
#include <array>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_buffer.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend {
class ShaderBindingTable final : public Object<IShaderBindingTable> {
 public:
  struct Region {
    uint64_t address{}, size{}, stride{};
  };

  ShaderBindingTable(IRayTracingProgram *program,
                     const ShaderBindingTableDesc &desc,
                     ObjectPtr<IBuffer> buffer,
                     const std::array<Region, 4> &regions)
      : program_(program),
        desc_(desc),
        buffer_(std::move(buffer)),
        regions_(regions) {
  }

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
};
}  // namespace grassland::rhi::backend
