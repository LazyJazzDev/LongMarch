#pragma once
#include "cao_di/vulkan/vulkan_util.h"

namespace CD::vulkan {

std::vector<const char *> GetValidationLayers();

bool CheckValidationLayerSupport();

VKAPI_ATTR VkBool32 VKAPI_CALL
DebugUtilsMessengerUserCallback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                VkDebugUtilsMessageTypeFlagsEXT message_type,
                                const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
                                void *user_data);
}  // namespace CD::vulkan
