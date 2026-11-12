#pragma once

#include <type_traits>

#include "grassland/rhi/rhi.h"
#ifdef LONGMARCH_VULKAN_ENABLED
#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"
#endif
#ifdef LONGMARCH_D3D12_ENABLED
#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"
#endif
#ifdef LONGMARCH_METAL_ENABLED
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#endif

namespace grassland::rhi {
// Dispatch tests to real native interfaces without a production common facet.
template <typename F>
void WithNativeBindingGroup(IBindingGroup *group, F &&test) {
#ifdef LONGMARCH_VULKAN_ENABLED
  if (auto native = QueryInterfacePtr<backend::vulkan::IVulkanBindingGroup>(group)) {
    test(native.Get());
    return;
  }
#endif
#ifdef LONGMARCH_D3D12_ENABLED
  if (auto native = QueryInterfacePtr<backend::d3d12::ID3D12BindingGroup>(group)) {
    test(native.Get());
    return;
  }
#endif
#ifdef LONGMARCH_METAL_ENABLED
  if (auto native = QueryInterfacePtr<backend::metal::IMetalBindingGroup>(group)) {
    test(native.Get());
    return;
  }
#endif
  FAIL() << "Binding group has no supported native interface";
}

template <typename T>
auto RetainBindingVersion(T *version) {
  return backend::ObjectPtr<T>::Retain(version);
}
}  // namespace grassland::rhi
