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
    case SHADER_TYPE_GEOMETRY:
      return VK_SHADER_STAGE_GEOMETRY_BIT;
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
    case RESOURCE_TYPE_SAMPLER:
      return VK_DESCRIPTOR_TYPE_SAMPLER;
    case RESOURCE_TYPE_ACCELERATION_STRUCTURE:
      return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
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

VkFilter FilterModeToVkFilter(FilterMode filter) {
  switch (filter) {
    case FILTER_MODE_NEAREST:
      return VK_FILTER_NEAREST;
    case FILTER_MODE_LINEAR:
      return VK_FILTER_LINEAR;
  }
  return VK_FILTER_NEAREST;
}

VkSamplerMipmapMode FilterModeToVkSamplerMipmapMode(FilterMode filter) {
  switch (filter) {
    case FILTER_MODE_NEAREST:
      return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    case FILTER_MODE_LINEAR:
      return VK_SAMPLER_MIPMAP_MODE_LINEAR;
  }
  return VK_SAMPLER_MIPMAP_MODE_NEAREST;
}

VkSamplerAddressMode AddressModeToVkSamplerAddressMode(AddressMode mode) {
  switch (mode) {
    case ADDRESS_MODE_REPEAT:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case ADDRESS_MODE_MIRRORED_REPEAT:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case ADDRESS_MODE_CLAMP_TO_EDGE:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case ADDRESS_MODE_CLAMP_TO_BORDER:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  }
  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkPrimitiveTopology PrimitiveTopologyToVkPrimitiveTopology(PrimitiveTopology topology) {
  switch (topology) {
    case PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case PRIMITIVE_TOPOLOGY_LINE_LIST:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case PRIMITIVE_TOPOLOGY_POINT_LIST:
      return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  }
  return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

VkBlendFactor BlendFactorToVkBlendFactor(BlendFactor factor) {
  switch (factor) {
    case BLEND_FACTOR_ZERO:
      return VK_BLEND_FACTOR_ZERO;
    case BLEND_FACTOR_ONE:
      return VK_BLEND_FACTOR_ONE;
    case BLEND_FACTOR_SRC_COLOR:
      return VK_BLEND_FACTOR_SRC_COLOR;
    case BLEND_FACTOR_ONE_MINUS_SRC_COLOR:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case BLEND_FACTOR_DST_COLOR:
      return VK_BLEND_FACTOR_DST_COLOR;
    case BLEND_FACTOR_ONE_MINUS_DST_COLOR:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case BLEND_FACTOR_SRC_ALPHA:
      return VK_BLEND_FACTOR_SRC_ALPHA;
    case BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case BLEND_FACTOR_DST_ALPHA:
      return VK_BLEND_FACTOR_DST_ALPHA;
    case BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
  }
  return VK_BLEND_FACTOR_ZERO;
}

VkBlendOp BlendOpToVkBlendOp(BlendOp op) {
  switch (op) {
    case BLEND_OP_ADD:
      return VK_BLEND_OP_ADD;
    case BLEND_OP_SUBTRACT:
      return VK_BLEND_OP_SUBTRACT;
    case BLEND_OP_REVERSE_SUBTRACT:
      return VK_BLEND_OP_REVERSE_SUBTRACT;
    case BLEND_OP_MIN:
      return VK_BLEND_OP_MIN;
    case BLEND_OP_MAX:
      return VK_BLEND_OP_MAX;
  }
  return VK_BLEND_OP_ADD;
}

VkPipelineColorBlendAttachmentState BlendStateToVkPipelineColorBlendAttachmentState(const BlendState &state) {
  VkPipelineColorBlendAttachmentState attachment{};
  attachment.blendEnable = state.blend_enable;
  attachment.srcColorBlendFactor = BlendFactorToVkBlendFactor(state.src_color);
  attachment.dstColorBlendFactor = BlendFactorToVkBlendFactor(state.dst_color);
  attachment.colorBlendOp = BlendOpToVkBlendOp(state.color_op);
  attachment.srcAlphaBlendFactor = BlendFactorToVkBlendFactor(state.src_alpha);
  attachment.dstAlphaBlendFactor = BlendFactorToVkBlendFactor(state.dst_alpha);
  attachment.alphaBlendOp = BlendOpToVkBlendOp(state.alpha_op);
  attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  return attachment;
}

VkPipelineBindPoint BindPointToVkPipelineBindPoint(BindPoint point) {
  switch (point) {
    case BIND_POINT_GRAPHICS:
      return VK_PIPELINE_BIND_POINT_GRAPHICS;
    case BIND_POINT_RAYTRACING:
      return VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
    default:
      throw std::runtime_error("Invalid bind point");
  }
}

VulkanResourceBinding::VulkanResourceBinding() : buffer(nullptr), image(nullptr) {
}

VulkanResourceBinding::VulkanResourceBinding(VulkanBuffer *buffer) : buffer(buffer), image(nullptr) {
}

VulkanResourceBinding::VulkanResourceBinding(VulkanImage *image) : buffer(nullptr), image(image) {
}

}  // namespace grassland::graphics::backend
