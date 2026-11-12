#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi::backend {
struct MappedBinding {
  BindingDesc desc;
  uint32_t group{}, shader_register{}, table_offset{};
};

std::vector<MappedBinding> MapBindings(const PipelineLayoutDesc &layout, BackendAPI api);
bool ValidateBindingLayout(const BindingLayoutDesc &layout);
bool SameBindingLayout(const BindingLayoutDesc &a, const BindingLayoutDesc &b);
}  // namespace grassland::rhi::backend
