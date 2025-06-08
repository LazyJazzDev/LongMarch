#include "grassland/vulkan/semaphore.h"

#ifdef _WIN32
#include <VersionHelpers.h>
#endif

namespace grassland::vulkan {
Semaphore::Semaphore(const struct Device *device, VkSemaphore semaphore) : device_(device), semaphore_(semaphore) {
}

Semaphore::~Semaphore() {
  vkDestroySemaphore(device_->Handle(), semaphore_, nullptr);
}
#if defined(LONGMARCH_CUDA_RUNTIME)
VkExternalSemaphoreHandleTypeFlagBits GetDefaultExternalSemaphoreHandleType() {
#ifdef _WIN32
  return IsWindows8OrGreater() ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
                               : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */
}
#endif

}  // namespace grassland::vulkan
