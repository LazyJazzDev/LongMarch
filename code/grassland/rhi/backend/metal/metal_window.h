#pragma once
#include <QuartzCore/QuartzCore.hpp>

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalWindow,
                        grassland::rhi::backend::IBackendWindow,
                        "e074a51b-c30d-43f4-8b6c-2d8fd1540424")

namespace grassland::rhi::backend::metal {

class IMetalWindow : public IBackendWindow {
 public:
  static rhi::IWindow *Create(IMetalDevice *device, const WindowDesc &desc);

  virtual bool IsFrameValid(uint64_t generation) const = 0;
};

}  // namespace grassland::rhi::backend::metal
