#pragma once
#include "grassland/graphics/interface.h"
#include "grassland/vulkan/vulkan.h"

namespace grassland::graphics::backend {

VkFormat ImageFormatToVkFormat(ImageFormat format);

VkFormat InputTypeToVkFormat(InputType type);

VkShaderStageFlagBits ShaderTypeToVkShaderStageFlags(ShaderType type);

class VulkanCore;
class VulkanBuffer;
class VulkanImage;
class VulkanShader;
class VulkanProgram;
class VulkanCommandContext;
class VulkanWindow;

}  // namespace grassland::graphics::backend
