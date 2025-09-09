#include "cao_di/vulkan/instance.h"

#include <utility>

#include "cao_di/vulkan/device.h"
#include "cao_di/vulkan/physical_device.h"
#include "cao_di/vulkan/surface.h"
#include "cao_di/vulkan/validation_layer.h"

namespace CD::vulkan {
InstanceCreateHint::InstanceCreateHint() {
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "CaoDi";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "CaoDi";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;

  ApplyGLFWSurfaceSupport();
}

void InstanceCreateHint::SetValidationLayersEnabled(bool enabled) {
  enable_validation_layers = enabled;
}

void InstanceCreateHint::AddExtension(const char *extension) {
  bool found = false;
  for (const auto &ext : extensions) {
    if (strcmp(ext, extension) == 0) {
      found = true;
      break;
    }
  }
  if (!found) {
    extensions.push_back(extension);
  }
}

bool InstanceCreateHint::IsEnabledExtension(const char *extension) const {
  for (auto ext : extensions) {
    if (std::strcmp(ext, extension) == 0) {
      return true;
    }
  }
  return false;
}

void InstanceCreateHint::ApplyGLFWSurfaceSupport() {
  uint32_t glfw_extension_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
  bool local_init = false;

  if (!glfw_extensions) {
    int err = glfwGetError(nullptr);
    if (err == GLFW_NOT_INITIALIZED) {
      glfwInit();
      glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
      local_init = true;
    }
  }

  if (glfw_extensions) {
    for (uint32_t i = 0; i < glfw_extension_count; i++) {
      AddExtension(glfw_extensions[i]);
    }
  }

  AddExtension(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);

  if (local_init) {
    glfwTerminate();
  }
}

VkResult CreateInstance(InstanceCreateHint create_hint, double_ptr<Instance> pp_instance) {
  VkInstanceCreateInfo instance_create_info{};
  VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};

  auto validation_layers = GetValidationLayers();
  if (create_hint.enable_validation_layers) {
    if (!CheckValidationLayerSupport()) {
      SetErrorMessage("validation layer is required, but not supported.");
      return VK_ERROR_UNKNOWN;
    }
    create_hint.AddExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instance_create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
    instance_create_info.ppEnabledLayerNames = validation_layers.data();
    instance_create_info.pNext = &debug_create_info;

    debug_create_info = {};
    debug_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_create_info.pfnUserCallback = DebugUtilsMessengerUserCallback;
  } else {
    instance_create_info.enabledLayerCount = 0;
    instance_create_info.pNext = nullptr;
  }

#ifdef __APPLE__
  create_hint.AddExtension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

  instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_create_info.pApplicationInfo = &create_hint.app_info;
  instance_create_info.enabledExtensionCount = static_cast<uint32_t>(create_hint.extensions.size());
  instance_create_info.ppEnabledExtensionNames = create_hint.extensions.data();

#ifdef __APPLE__
  instance_create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

  VkInstance instance{nullptr};
  VkDebugUtilsMessengerEXT debug_messenger{nullptr};
  InstanceProcedures instance_procedures;

  RETURN_IF_FAILED_VK(vkCreateInstance(&instance_create_info, nullptr, &instance), "failed to create instance.");

  instance_procedures.Initialize(instance, create_hint.enable_validation_layers);

  if (create_hint.enable_validation_layers) {
    RETURN_IF_FAILED_VK(
        instance_procedures.vkCreateDebugUtilsMessengerEXT(instance, &debug_create_info, nullptr, &debug_messenger),
        "failed to construct up debug messenger.");
  }

  if (pp_instance) {
    pp_instance.construct(create_hint, instance, debug_messenger, instance_procedures);
  } else {
    if (create_hint.enable_validation_layers) {
      instance_procedures.vkDestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
    SetErrorMessage("pp_instance is nullptr.");
    return VK_ERROR_UNKNOWN;
  }

  return VK_SUCCESS;
}

Instance::Instance(InstanceCreateHint create_hint,
                   VkInstance instance,
                   VkDebugUtilsMessengerEXT debug_messenger,
                   InstanceProcedures instance_procedures)
    : create_hint_(std::move(create_hint)),
      instance_(instance),
      debug_messenger_(debug_messenger),
      instance_procedures_(instance_procedures) {
}

Instance::~Instance() {
  if (create_hint_.enable_validation_layers) {
    instance_procedures_.vkDestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
  }

  vkDestroyInstance(instance_, nullptr);
}

VkResult Instance::CreateSurfaceFromGLFWWindow(GLFWwindow *window, double_ptr<Surface> pp_surface) const {
  VkSurfaceKHR surface{nullptr};
  VkResult result = glfwCreateWindowSurface(instance_, window, nullptr, &surface);
  if (result != VK_SUCCESS) {
    SetErrorMessage("failed to create window surface.");
    return result;
  }

  if (pp_surface) {
    pp_surface.construct(this, window, surface);
  } else {
    vkDestroySurfaceKHR(instance_, surface, nullptr);
    SetErrorMessage("pp_surface is nullptr.");
    return VK_ERROR_UNKNOWN;
  }

  return VK_SUCCESS;
}

std::vector<PhysicalDevice> Instance::EnumeratePhysicalDevices() const {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

  std::vector<PhysicalDevice> physical_devices;
  physical_devices.reserve(devices.size());

  for (const auto &device : devices) {
    physical_devices.emplace_back(device);
  }

  return physical_devices;
}

VkResult Instance::CreateDevice(Surface *surface,
                                bool enable_raytracing_extension,
                                int device_index,
                                double_ptr<struct Device> pp_device) const {
  DeviceFeatureRequirement device_feature_requirement{};
  device_feature_requirement.enable_raytracing_extension = enable_raytracing_extension;
  return CreateDevice(device_feature_requirement, device_index, pp_device);
}

VkResult Instance::CreateDevice(const struct DeviceFeatureRequirement &device_feature_requirement,
                                int device_index,
                                double_ptr<struct Device> pp_device) const {
  std::vector<PhysicalDevice> physical_devices = EnumeratePhysicalDevices();

  if (device_index < 0) {
    uint64_t max_score = 0;
    for (int i = 0; i < physical_devices.size(); i++) {
      if (!physical_devices[i].CheckFeatureSupport(device_feature_requirement)) {
        continue;
      }

      uint64_t score = physical_devices[i].Evaluate();
      if (device_index < 0 || score > max_score) {
        max_score = score;
        device_index = i;
      }
    }
  }

  if (device_index < 0 || device_index >= physical_devices.size()) {
    SetErrorMessage("no suitable physical device found.");
    return VK_ERROR_UNKNOWN;
  }

  PhysicalDevice physical_device = physical_devices[device_index];

  auto create_info = device_feature_requirement.GenerateRecommendedDeviceCreateInfo(physical_device);

  if (create_hint_.IsEnabledExtension(VK_KHR_SURFACE_EXTENSION_NAME) &&
      physical_device.IsExtensionSupported(VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
    create_info.AddExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
  VkResult result = CreateDevice(physical_device, device_feature_requirement, create_info, pp_device);
  if (result == VK_SUCCESS) {
    if (device_feature_requirement.enable_raytracing_extension) {
      pp_device->Procedures().GetRayTracingProcedures(pp_device->Handle());
    }
  }

  return result;
}

VkResult Instance::CreateDevice(const PhysicalDevice &physical_device,
                                const DeviceFeatureRequirement &device_feature_requirement,
                                DeviceCreateInfo create_info,
                                double_ptr<Device> pp_device) const {
  return CreateDevice(physical_device, create_info, device_feature_requirement.GetVmaAllocatorCreateFlags(), pp_device);
}

VkResult Instance::CreateDevice(const PhysicalDevice &physical_device,
                                struct DeviceCreateInfo create_info,
                                VmaAllocatorCreateFlags allocator_flags,
                                double_ptr<class Device> pp_device) const {
  VkDeviceCreateInfo device_create_info =
      create_info.CompileVkDeviceCreateInfo(create_hint_.enable_validation_layers, physical_device);

  VkDevice device{nullptr};

  RETURN_IF_FAILED_VK(vkCreateDevice(physical_device.Handle(), &device_create_info, nullptr, &device),
                      "failed to create logical device.");

  if (pp_device) {
    pp_device.construct(this, physical_device, create_info, allocator_flags, device);
  } else {
    vkDestroyDevice(device, nullptr);
    SetErrorMessage("pp_device is nullptr.");
    return VK_ERROR_UNKNOWN;
  }

  return VK_SUCCESS;
}

VkResult Instance::CreateDevice(Surface *surface,
                                bool enable_raytracing_extension,
                                double_ptr<struct Device> pp_device) const {
  return CreateDevice(surface, enable_raytracing_extension, -1, pp_device);
}

VkResult Instance::CreateDevice(const DeviceFeatureRequirement &device_feature_requirement,
                                double_ptr<struct Device> pp_device) const {
  return CreateDevice(device_feature_requirement, -1, pp_device);
}

}  // namespace CD::vulkan
