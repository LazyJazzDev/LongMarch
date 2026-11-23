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

struct BindingImageUse {
  ImageView view;
  bool writable{};
};

// Disjoint mips may be read and written by the same dispatch. Overlapping writes
// through multiple bindings are rejected before drawing/dispatching.
bool AddBindingImageUse(std::vector<BindingImageUse> &uses, const ImageView &view, bool writable);
bool NormalizeImageView(ImageView &view, BindingType type);
bool SameBindingLayout(const BindingLayoutDesc &a, const BindingLayoutDesc &b);
}  // namespace grassland::rhi::backend
