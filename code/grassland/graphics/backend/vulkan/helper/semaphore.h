#pragma once

#include "grassland/graphics/backend/vulkan/helper/device.h"
#include "grassland/graphics/backend/vulkan/helper/vulkan_util.h"

namespace grassland::graphics::backend::vulkan {
class Semaphore {
 public:
  Semaphore(const class Device *device, VkSemaphore semaphore);

  ~Semaphore();

  VkSemaphore Handle() const {
    return semaphore_;
  }

  const class Device *Device() const {
    return device_;
  }

 private:
  const class Device *device_{};
  VkSemaphore semaphore_{};
};

#if defined(LONGMARCH_CUDA_RUNTIME)
VkExternalSemaphoreHandleTypeFlagBits GetDefaultExternalSemaphoreHandleType();
#endif

}  // namespace grassland::graphics::backend::vulkan
