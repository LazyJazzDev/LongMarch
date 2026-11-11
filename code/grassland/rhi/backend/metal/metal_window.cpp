#include "grassland/rhi/backend/metal/metal_window.h"

#include "grassland/rhi/backend/metal/metal_window_impl.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalWindow, "e074a51b-c30d-43f4-8b6c-2d8fd1540424")

namespace grassland::rhi::backend::metal {
IMetalWindow *IMetalWindow::Create(IMetalDevice *device, const WindowDesc &desc) {
  return MetalWindowImpl::Create(device, desc);
}
}  // namespace grassland::rhi::backend::metal
