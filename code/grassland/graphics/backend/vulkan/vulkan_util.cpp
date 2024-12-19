#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {
VkFormat ImageFormatToVkFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return VK_FORMAT_B8G8R8A8_UNORM;
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      return VK_FORMAT_R32G32B32_SFLOAT;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return VK_FORMAT_R32G32_SFLOAT;
    case IMAGE_FORMAT_R32_SFLOAT:
      return VK_FORMAT_R32_SFLOAT;
    case IMAGE_FORMAT_D32_SFLOAT:
      return VK_FORMAT_D32_SFLOAT;
    default:
      return VK_FORMAT_UNDEFINED;
  }
}

VkFormat InputTypeToVkFormat(InputType type) {
  switch (type) {
    case INPUT_TYPE_UINT:
      return VK_FORMAT_R32_UINT;
    case INPUT_TYPE_INT:
      return VK_FORMAT_R32_SINT;
    case INPUT_TYPE_FLOAT:
      return VK_FORMAT_R32_SFLOAT;
    case INPUT_TYPE_UINT2:
      return VK_FORMAT_R32G32_UINT;
    case INPUT_TYPE_INT2:
      return VK_FORMAT_R32G32_SINT;
    case INPUT_TYPE_FLOAT2:
      return VK_FORMAT_R32G32_SFLOAT;
    case INPUT_TYPE_UINT3:
      return VK_FORMAT_R32G32B32_UINT;
    case INPUT_TYPE_INT3:
      return VK_FORMAT_R32G32B32_SINT;
    case INPUT_TYPE_FLOAT3:
      return VK_FORMAT_R32G32B32_SFLOAT;
    case INPUT_TYPE_UINT4:
      return VK_FORMAT_R32G32B32A32_UINT;
    case INPUT_TYPE_INT4:
      return VK_FORMAT_R32G32B32A32_SINT;
    case INPUT_TYPE_FLOAT4:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
    default:
      return VK_FORMAT_UNDEFINED;
  }
}

VkShaderStageFlagBits ShaderTypeToVkShaderStageFlags(ShaderType type) {
  switch (type) {
    case SHADER_TYPE_VERTEX:
      return VK_SHADER_STAGE_VERTEX_BIT;
    case SHADER_TYPE_FRAGMENT:
      return VK_SHADER_STAGE_FRAGMENT_BIT;
    default:
      return VK_SHADER_STAGE_ALL;
  }
}

VkDescriptorType ResourceTypeToVkDescriptorType(ResourceType type) {
  switch (type) {
    case RESOURCE_TYPE_IMAGE:
      return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case RESOURCE_TYPE_TEXTURE:
      return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case RESOURCE_TYPE_UNIFORM_BUFFER:
      return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case RESOURCE_TYPE_STORAGE_BUFFER:
      return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    default:
      return VK_DESCRIPTOR_TYPE_MAX_ENUM;
  }
}

VkCullModeFlagBits CullModeToVkCullMode(CullMode mode) {
  switch (mode) {
    case CULL_MODE_NONE:
      return VK_CULL_MODE_NONE;
    case CULL_MODE_FRONT:
      return VK_CULL_MODE_FRONT_BIT;
    case CULL_MODE_BACK:
      return VK_CULL_MODE_BACK_BIT;
    default:
      return VK_CULL_MODE_NONE;
  }
}

VulkanResourceBinding::VulkanResourceBinding()
    : buffer(nullptr), image(nullptr) {
}

VulkanResourceBinding::VulkanResourceBinding(VulkanBuffer *buffer)
    : buffer(buffer), image(nullptr) {
}

VulkanResourceBinding::VulkanResourceBinding(VulkanImage *image)
    : buffer(nullptr), image(image) {
}

}  // namespace grassland::graphics::backend
