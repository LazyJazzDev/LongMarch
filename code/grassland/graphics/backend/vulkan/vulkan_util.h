#pragma once
#include "grassland/graphics/interface.h"
#include "grassland/vulkan/vulkan.h"

namespace grassland::graphics::backend {

VkFormat ImageFormatToVkFormat(ImageFormat format);

VkFormat InputTypeToVkFormat(InputType type);

VkShaderStageFlagBits ShaderTypeToVkShaderStageFlags(ShaderType type);

VkDescriptorType ResourceTypeToVkDescriptorType(ResourceType type);

VkCullModeFlagBits CullModeToVkCullMode(CullMode mode);

VkFilter FilterModeToVkFilter(FilterMode filter);

VkSamplerMipmapMode FilterModeToVkSamplerMipmapMode(FilterMode filter);

VkSamplerAddressMode AddressModeToVkSamplerAddressMode(AddressMode mode);

VkPrimitiveTopology PrimitiveTopologyToVkPrimitiveTopology(
    PrimitiveTopology topology);

class VulkanCore;
class VulkanBuffer;
class VulkanImage;
class VulkanSampler;
class VulkanShader;
class VulkanProgram;
class VulkanCommandContext;
class VulkanWindow;

struct VulkanResourceBinding {
  VulkanResourceBinding();

  VulkanResourceBinding(VulkanBuffer *buffer);

  VulkanResourceBinding(VulkanImage *image);

  VulkanBuffer *buffer;
  VulkanImage *image;
};

}  // namespace grassland::graphics::backend
