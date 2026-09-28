#include "grassland/graphics/backend/vulkan/helper/instance.h"

#include <utility>

#include "grassland/graphics/backend/vulkan/helper/physical_device.h"
#include "grassland/graphics/backend/vulkan/helper/validation_layer.h"

namespace grassland::graphics::backend::vulkan {
InstanceCreateHint::InstanceCreateHint() {
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Grassland";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "Grassland";
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

VkResult CreateNativeInstance(InstanceCreateHint &create_hint,
                              VkInstance &instance,
                              VkDebugUtilsMessengerEXT &debug_messenger,
                              InstanceProcedures &instance_procedures) {
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

  RETURN_IF_FAILED_VK(vkCreateInstance(&instance_create_info, nullptr, &instance), "failed to create instance.");

  LoadInstanceProcedures(instance, create_hint.enable_validation_layers, instance_procedures);

  if (create_hint.enable_validation_layers) {
    RETURN_IF_FAILED_VK(
        instance_procedures.vkCreateDebugUtilsMessengerEXT(instance, &debug_create_info, nullptr, &debug_messenger),
        "failed to construct up debug messenger.");
  }

  return VK_SUCCESS;
}

std::vector<VkPhysicalDevice> EnumerateNativePhysicalDevices(VkInstance instance) {
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(instance, &count, nullptr);
  std::vector<VkPhysicalDevice> handles(count);
  vkEnumeratePhysicalDevices(instance, &count, handles.data());
  return handles;
}

}  // namespace grassland::graphics::backend::vulkan
