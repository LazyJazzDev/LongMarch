#include "grassland/vulkan/validation_layer.h"

#include "iostream"

namespace grassland::vulkan {

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

bool CheckValidationLayerSupport() {
  uint32_t layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

  std::vector<VkLayerProperties> available_layers(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

  for (const char *layer_name : validationLayers) {
    bool layer_found = false;

    for (const auto &layer_properties : available_layers) {
      if (std::strcmp(layer_name, layer_properties.layerName) == 0) {
        layer_found = true;
        break;
      }
    }

    if (!layer_found) {
      return false;
    }
  }

  return true;
}

VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsMessengerUserCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
    void *user_data) {
  if (message_severity <= VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
    return VK_FALSE;
  }
  std::string message_tag;
  auto add_tag = [&message_tag](const char *tag) {
    if (!message_tag.empty()) {
      message_tag += ", ";
    }
    message_tag += tag;
  };
  if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    add_tag("ERROR");
  }
  if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    add_tag("WARNING");
  }
  if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    add_tag("INFO");
  }
  if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
    add_tag("VERBOSE");
  }
  std::cerr << fmt::format("validation layer ({}): {}", message_tag,
                           callback_data->pMessage)
            << std::endl;

  if (message_severity > VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    throw std::runtime_error("validation layer error");
  }
  return VK_FALSE;
}

std::vector<const char *> GetValidationLayers() {
  return validationLayers;
}
}  // namespace grassland::vulkan
