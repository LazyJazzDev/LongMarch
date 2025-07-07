#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define VMA_VULKAN_VERSION 1002000  // Specify using Vulkan 1.2
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#ifdef _WIN64
// clang-format off
#define NOMINMAX
#include <Windows.h>

#include "vulkan/vulkan_win32.h"
// clang-format on
#endif

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "grassland/graphics/graphics_util.h"
#include "grassland/util/util.h"

namespace grassland::vulkan {
#define GRASSLAND_VULKAN_PROCEDURE_VAR(function_name) PFN_##function_name function_name{};

#ifdef NDEBUG
constexpr bool kDefaultEnableValidationLayers = false;
#else
constexpr bool kDefaultEnableValidationLayers = true;
#endif

std::string VkFormatToName(VkFormat format);

std::string VkColorSpaceToName(VkColorSpaceKHR color_space);

std::string VkPresentModeToName(VkPresentModeKHR present_mode);

void ThrowError(const std::string &message);

template <class... Args>
void ThrowError(const std::string &message, Args &&...args) {
  ThrowError(fmt::format(message, std::forward<Args>(args)...));
}

void ThrowIfFailed(VkResult result, const std::string &message);

template <class... Args>
void ThrowIfFailed(VkResult result, const std::string &message, Args &&...args) {
  ThrowIfFailed(result, fmt::format(message, std::forward<Args>(args)...));
}

void Warning(const std::string &message);

template <class... Args>
void Warning(const std::string &message, Args &&...args) {
  Warning(fmt::format(message, std::forward<Args>(args)...));
}

void SetErrorMessage(const std::string &message);

template <class... Args>
void SetErrorMessage(const std::string &message, Args &&...args) {
  SetErrorMessage(fmt::format(message, std::forward<Args>(args)...));
}

std::string GetErrorMessage();

std::string VkResultToString(VkResult result);

#define RETURN_IF_FAILED_VK(cmd, ...)                                                                   \
  do {                                                                                                  \
    VkResult res = cmd;                                                                                 \
    if (res != VK_SUCCESS) {                                                                            \
      ::grassland::vulkan::SetErrorMessage(__VA_ARGS__);                                                \
      ::grassland::vulkan::SetErrorMessage("VkResult: {}", ::grassland::vulkan::VkResultToString(res)); \
      return res;                                                                                       \
    }                                                                                                   \
                                                                                                        \
  } while (false)

class Instance;
class Surface;
class PhysicalDevice;
class Device;
class Queue;
class Swapchain;
class CommandPool;
class CommandBuffer;
class DescriptorPool;
class DescriptorSetLayout;
class DescriptorSet;
class PipelineLayout;
class Pipeline;
class RenderPass;
class Framebuffer;
class ShaderModule;
struct HitGroup;
class Fence;
class Semaphore;
class Buffer;
class Image;
class Sampler;
class AccelerationStructure;
class RayTracingPipeline;
class ShaderBindingTable;

bool IsDepthFormat(VkFormat format);

using graphics::CompiledShaderBlob;

}  // namespace grassland::vulkan
