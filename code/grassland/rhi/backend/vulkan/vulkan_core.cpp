#include "grassland/rhi/backend/vulkan/vulkan_core.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/util/log.h"

namespace grassland::rhi::backend::vulkan {

VulkanCore::VulkanCore() = default;

VkInstance VulkanCore::Instance() const {
  return instance_;
}

VulkanCore *VulkanCore::CreateVulkanCore(const CoreDesc &desc) {
  uint32_t loader_version = VK_API_VERSION_1_0;
  const auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
      vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
  if (enumerate_version && !Check(enumerate_version(&loader_version), "vkEnumerateInstanceVersion"))
    return {};
  if (loader_version < VK_API_VERSION_1_2) {
    ReportError(-1, "RHI requires Vulkan 1.2 or newer.");
    return {};
  }
  uint32_t supported_version = VK_API_VERSION_1_2;
#ifdef VK_VERSION_1_3
  supported_version = VK_API_VERSION_1_3;
#endif
#ifdef VK_VERSION_1_4
  supported_version = VK_API_VERSION_1_4;
#endif
  const uint32_t api_version = std::min(loader_version, supported_version);
  std::vector<VkExtensionProperties> available;
  if (!EnumerateList(
          [](uint32_t *count, VkExtensionProperties *data) {
            return vkEnumerateInstanceExtensionProperties(nullptr, count, data);
          },
          available))
    return {};
  std::vector<const char *> extensions;
  VkInstanceCreateFlags flags = 0;
  for (const auto &extension : available) {
    for (const auto name :
         {"VK_KHR_surface", "VK_KHR_win32_surface", "VK_KHR_xlib_surface", "VK_KHR_xcb_surface",
          "VK_KHR_wayland_surface", "VK_EXT_swapchain_colorspace", "VK_EXT_metal_surface", "VK_MVK_macos_surface"}) {
      if (std::strcmp(extension.extensionName, name) == 0)
        extensions.push_back(name);
    }
    if (std::strcmp(extension.extensionName, "VK_KHR_portability_enumeration") == 0) {
      extensions.push_back("VK_KHR_portability_enumeration");
      flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
  }
  const char *validation_layer = "VK_LAYER_KHRONOS_validation";
  VkDebugUtilsMessengerCreateInfoEXT debug{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  if (desc.enable_debug) {
    std::vector<VkLayerProperties> layers;
    if (!EnumerateList([](uint32_t *n, VkLayerProperties *p) { return vkEnumerateInstanceLayerProperties(n, p); },
                       layers))
      return nullptr;
    if (std::none_of(layers.begin(), layers.end(),
                     [&](const auto &layer) { return std::strcmp(layer.layerName, validation_layer) == 0; }) ||
        std::none_of(available.begin(), available.end(), [](const auto &extension) {
          return std::strcmp(extension.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
        })) {
      ReportError(-1, "Requested Vulkan validation layer or debug-utils extension is unavailable.");
      return nullptr;
    }
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    debug.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                               const VkDebugUtilsMessengerCallbackDataEXT *data, void *) -> VkBool32 {
      if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        LogError(data->pMessage);
      else
        LogWarning(data->pMessage);
      return VK_FALSE;
    };
  }
  VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application.pApplicationName = "LongMarch RHI";
  application.apiVersion = api_version;
  VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  create.flags = flags;
  if (desc.enable_debug) {
    create.enabledLayerCount = 1;
    create.ppEnabledLayerNames = &validation_layer;
    create.pNext = &debug;
  }
  create.pApplicationInfo = &application;
  create.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create.ppEnabledExtensionNames = extensions.data();
  auto core = std::unique_ptr<VulkanCore>(new VulkanCore());
  if (!Check(vkCreateInstance(&create, nullptr, &core->instance_), "vkCreateInstance"))
    return {};
  if (desc.enable_debug) {
    auto create_debug = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(core->instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (!create_debug || !Check(create_debug(core->instance_, &debug, nullptr, &core->debug_messenger_),
                                "Create Vulkan debug messenger"))
      return nullptr;
  }
  std::vector<VkPhysicalDevice> devices;
  if (!EnumerateList([&](uint32_t *count,
                         VkPhysicalDevice *data) { return vkEnumeratePhysicalDevices(core->instance_, count, data); },
                     devices))
    return {};
  for (auto device : devices) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device, &properties);
    if (properties.apiVersion < VK_API_VERSION_1_2)
      continue;
    const uint32_t device_version = std::min(api_version, properties.apiVersion);
    DeviceFeatures features(device, device_version);
    if (!features)
      return {};
    if (!features.TimelineSemaphore())
      continue;
    PhysicalDevice info{};
    info.name = properties.deviceName;
    switch (properties.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        info.type = PHYSICAL_DEVICE_TYPE_DEDICATED;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        info.type = PHYSICAL_DEVICE_TYPE_INTEGRATED;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:
        info.type = PHYSICAL_DEVICE_TYPE_SOFTWARE;
        break;
      default:
        info.type = PHYSICAL_DEVICE_TYPE_OTHER;
        break;
    }
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(device, &memory);
    for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
      if (info.type == PHYSICAL_DEVICE_TYPE_DEDICATED &&
          (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
        info.dedicated_memory += memory.memoryHeaps[i].size;
      } else {
        info.unified_memory += memory.memoryHeaps[i].size;
      }
    }
    info.geometry_shader_feature = features.Features().features.geometryShader == VK_TRUE;
    info.ray_tracing_pipeline_feature = features.RayTracingPipeline();
    info.ray_query_feature = features.RayQuery();
    core->adapters_.push_back({device, device_version, std::move(info)});
  }
  return core.release();
}

VulkanCore::~VulkanCore() {
  if (debug_messenger_) {
    auto destroy_debug = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
    destroy_debug(instance_, debug_messenger_, nullptr);
  }
  if (instance_)
    vkDestroyInstance(instance_, nullptr);
}

std::vector<PhysicalDevice> VulkanCore::GetPhysicalDevices() const {
  std::vector<PhysicalDevice> devices;
  for (const auto &adapter : adapters_)
    devices.push_back(adapter.info);
  return devices;
}

rhi::IDevice *VulkanCore::CreateDevice(int32_t device_idx) {
#if !defined(NDEBUG)
  if (device_idx < 0 || static_cast<size_t>(device_idx) >= adapters_.size()) {
    ReportError(-1, "Physical device ID is out of range.");
    return {};
  }
#endif
  const auto &adapter = adapters_[device_idx];
  return VulkanDevice::CreateVulkanDevice(adapter.info, this, adapter.native, adapter.api_version);
}

}  // namespace grassland::rhi::backend::vulkan
