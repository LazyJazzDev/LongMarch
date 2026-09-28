#include "grassland/graphics/backend/vulkan/helper/semaphore.h"

#ifdef _WIN64
#include <VersionHelpers.h>
#endif

namespace grassland::graphics::backend::vulkan {
#if defined(LONGMARCH_CUDA_RUNTIME)
VkExternalSemaphoreHandleTypeFlagBits GetDefaultExternalSemaphoreHandleType() {
#ifdef _WIN64
  return IsWindows8OrGreater() ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
                               : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */
}
#endif

}  // namespace grassland::graphics::backend::vulkan
