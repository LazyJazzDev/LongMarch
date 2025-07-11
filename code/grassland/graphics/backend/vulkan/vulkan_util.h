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

VkPrimitiveTopology PrimitiveTopologyToVkPrimitiveTopology(PrimitiveTopology topology);

VkBlendFactor BlendFactorToVkBlendFactor(BlendFactor factor);

VkBlendOp BlendOpToVkBlendOp(BlendOp op);

VkPipelineColorBlendAttachmentState BlendStateToVkPipelineColorBlendAttachmentState(const BlendState &state);

VkPipelineBindPoint BindPointToVkPipelineBindPoint(BindPoint point);

VkAccelerationStructureInstanceKHR RayTracingInstanceToVkAccelerationStructureInstanceKHR(
    const RayTracingInstance &instance);

class VulkanCore;
class VulkanBuffer;
class VulkanImage;
class VulkanSampler;
class VulkanShader;
class VulkanProgram;
class VulkanComputeProgram;
class VulkanCommandContext;
class VulkanWindow;
class VulkanAccelerationStructure;
class VulkanRayTracingProgram;
class VulkanProgramBase;

struct VulkanResourceBinding {
  VulkanResourceBinding();

  VulkanResourceBinding(VulkanBuffer *buffer);

  VulkanResourceBinding(VulkanImage *image);

  VulkanBuffer *buffer;
  VulkanImage *image;
};

#if defined(LONGMARCH_CUDA_RUNTIME)
class VulkanCUDABuffer;
#endif

}  // namespace grassland::graphics::backend
