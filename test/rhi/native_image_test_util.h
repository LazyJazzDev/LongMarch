#pragma once
#include "grassland/rhi/rhi.h"
#ifdef LONGMARCH_VULKAN_ENABLED
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_window.h"
#endif
#ifdef LONGMARCH_D3D12_ENABLED
#include "grassland/rhi/backend/d3d12/d3d12_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_window.h"
#endif
#ifdef LONGMARCH_METAL_ENABLED
#include "grassland/rhi/backend/metal/metal_image.h"
#include "grassland/rhi/backend/metal/metal_window.h"
#endif
namespace grassland::rhi {
inline bool NativeImageSupportsStorage(IImage *image) {
#ifdef LONGMARCH_VULKAN_ENABLED
  if (auto native = QueryInterfacePtr<backend::vulkan::IVulkanImage>(image))
    return native->SupportsStorage();
#endif
#ifdef LONGMARCH_D3D12_ENABLED
  if (auto native = QueryInterfacePtr<backend::d3d12::ID3D12Image>(image))
    return native->SupportsStorage();
#endif
#ifdef LONGMARCH_METAL_ENABLED
  if (auto native = QueryInterfacePtr<backend::metal::IMetalImage>(image))
    return native->SupportsStorage();
#endif
  return false;
}

inline bool RecordNativeImGui(IWindow *window, ICommandContext *commands, IImage *image) {
#ifdef LONGMARCH_VULKAN_ENABLED
  if (auto native = QueryInterfacePtr<backend::vulkan::IVulkanWindow>(window))
    return native->RecordImGui(commands, image);
#endif
#ifdef LONGMARCH_D3D12_ENABLED
  if (auto native = QueryInterfacePtr<backend::d3d12::ID3D12Window>(window))
    return native->RecordImGui(commands, image);
#endif
#ifdef LONGMARCH_METAL_ENABLED
  if (auto native = QueryInterfacePtr<backend::metal::IMetalWindow>(window))
    return native->RecordImGui(commands, image);
#endif
  return false;
}
}  // namespace grassland::rhi
