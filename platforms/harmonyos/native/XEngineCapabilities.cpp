#include "XEngineCapabilities.h"

#include <dlfcn.h>
#include <hilog/log.h>

#include <memory>
#include <vector>

#ifdef LONGMARCH_HAVE_XENGINE_HEADERS
#include <xengine/xeg_vulkan_extension.h>
#endif

namespace longmarch::harmony {
void LogXEngineCapabilities(VkPhysicalDevice physical_device) {
#ifdef LONGMARCH_HAVE_XENGINE_HEADERS
  // Load the public kit optionally so a device without it can still render.
  std::unique_ptr<void, decltype(&dlclose)> library(dlopen("libxengine.so", RTLD_NOW | RTLD_LOCAL), dlclose);
  if (!library) {
    OH_LOG_Print(LOG_APP, LOG_INFO, 0, "LongMarchGPU", "XEngine library unavailable: %{public}s", dlerror());
    return;
  }
  auto enumerate = reinterpret_cast<PFN_HMS_XEG_EnumerateDeviceExtensionProperties>(
      dlsym(library.get(), "HMS_XEG_EnumerateDeviceExtensionProperties"));
  if (!enumerate) {
    OH_LOG_Print(LOG_APP, LOG_INFO, 0, "LongMarchGPU", "XEngine extension enumeration unavailable");
    return;
  }
  uint32_t count = 0;
  VkResult result = enumerate(physical_device, &count, nullptr);
  if (result != VK_SUCCESS) {
    OH_LOG_Print(LOG_APP, LOG_INFO, 0, "LongMarchGPU", "XEngine count query returned %{public}d", result);
    return;
  }
  std::vector<XEG_ExtensionProperties> extensions(count);
  if (count)
    result = enumerate(physical_device, &count, extensions.data());
  OH_LOG_Print(LOG_APP, LOG_INFO, 0, "LongMarchGPU", "XEngine query result=%{public}d count=%{public}u", result, count);
  if (result != VK_SUCCESS && result != VK_INCOMPLETE)
    return;
  for (size_t i = 0; i < count && i < extensions.size(); ++i)
    OH_LOG_Print(LOG_APP, LOG_INFO, 0, "LongMarchGPU", "XEngine extension %{public}s version=%{public}u",
                 extensions[i].extensionName, extensions[i].version);
#else
  OH_LOG_Print(LOG_APP, LOG_INFO, 0, "LongMarchGPU", "XEngine SDK headers unavailable at build time");
#endif
}
}  // namespace longmarch::harmony
