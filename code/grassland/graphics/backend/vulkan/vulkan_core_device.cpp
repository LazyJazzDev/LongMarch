#include <numeric>
#include <utility>

#include "grassland/graphics/backend/vulkan/helper/buffer.h"
#include "grassland/graphics/backend/vulkan/helper/descriptor_pool.h"
#include "grassland/graphics/backend/vulkan/helper/image.h"
#include "grassland/graphics/backend/vulkan/helper/instance.h"
#include "grassland/graphics/backend/vulkan/helper/instance_procedures.h"
#include "grassland/graphics/backend/vulkan/helper/pipeline.h"
#include "grassland/graphics/backend/vulkan/helper/raytracing/raytracing.h"
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_shader.h"

namespace grassland::graphics::backend {
using namespace vulkan;

void VulkanCore::InitializeNativeDevice(uint32_t api_version,
                                        const vulkan::PhysicalDevice &physical_device,
                                        vulkan::DeviceCreateInfo create_info,
                                        VmaAllocatorCreateFlags allocator_flags,
                                        VkDevice device) {
  api_version_ = api_version;
  physical_device_ = physical_device;
  create_info_.emplace(std::move(create_info));
  device_ = device;
  VmaAllocatorCreateInfo allocator_info = {};
  allocator_info.physicalDevice = physical_device_->Handle();
  allocator_info.device = device_;
  allocator_info.instance = instance_;
  allocator_info.vulkanApiVersion = api_version_;
  allocator_info.flags = allocator_flags;
  vmaCreateAllocator(&allocator_info, &allocator_);

  bool ray_tracing_enabled = false;
  bool acceleration_structure_enabled = false;

  for (auto extension : create_info_->extensions) {
    if (strcmp(extension, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0) {
      ray_tracing_enabled = true;
    }
    if (strcmp(extension, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0)
      acceleration_structure_enabled = true;
  }

  if (ray_tracing_enabled) {
    procedures_.GetRayTracingProcedures(device_);
  } else if (acceleration_structure_enabled) {
    procedures_.GetAccelerationStructureProcedures(device_);
  }

  subgroup_properties_ = {};
  subgroup_properties_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &subgroup_properties_;
  vkGetPhysicalDeviceProperties2(physical_device_->Handle(), &properties2);
}

void VulkanCore::DestroyNativeDevice() {
  vmaDestroyAllocator(allocator_);
  vkDestroyDevice(device_, nullptr);
}

VkResult VulkanCore::CreateDescriptorPool(const std::vector<VkDescriptorPoolSize> &pool_sizes,
                                          uint32_t max_sets,
                                          VkDescriptorPool *pool) const {
  VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
  info.pPoolSizes = pool_sizes.data();
  info.maxSets = max_sets;
  return vkCreateDescriptorPool(device_, &info, nullptr, pool);
}

VkResult VulkanCore::CreateImage(VkFormat format,
                                 VkExtent2D extent,
                                 VkImageUsageFlags usage,
                                 VkImageAspectFlags aspect,
                                 VkSampleCountFlagBits sample_count,
                                 VmaMemoryUsage mem_usage,
                                 double_ptr<vulkan::Image> pp_image) const {
  if (!pp_image) {
    SetErrorMessage("pp_image is nullptr");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  VkImageCreateInfo image_create_info{};
  image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_create_info.imageType = VK_IMAGE_TYPE_2D;
  image_create_info.format = format;
  image_create_info.extent.width = extent.width;
  image_create_info.extent.height = extent.height;
  image_create_info.extent.depth = 1;
  image_create_info.mipLevels = 1;
  image_create_info.arrayLayers = 1;
  image_create_info.samples = sample_count;
  image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_create_info.usage = usage;
  image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  // Create image from image create info by VMA library

  VmaAllocationCreateInfo allocation_info = {};
  allocation_info.usage = mem_usage;

  VkImage image;
  VmaAllocation allocation;

  RETURN_IF_FAILED_VK(vmaCreateImage(allocator_, &image_create_info, &allocation_info, &image, &allocation, nullptr),
                      "failed to create image!");

  VkImageViewCreateInfo image_view_create_info{};
  image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_create_info.image = image;
  image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  image_view_create_info.format = format;
  image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.subresourceRange.aspectMask = aspect;
  image_view_create_info.subresourceRange.baseMipLevel = 0;
  image_view_create_info.subresourceRange.levelCount = 1;
  image_view_create_info.subresourceRange.baseArrayLayer = 0;
  image_view_create_info.subresourceRange.layerCount = 1;

  VkImageView image_view;

  RETURN_IF_FAILED_VK(vkCreateImageView(device_, &image_view_create_info, nullptr, &image_view),
                      "failed to create image view!");

  pp_image.construct(this, format, extent, usage, aspect, sample_count, image, image_view, allocation);

  return VK_SUCCESS;
}

VkResult VulkanCore::CreateImage(VkFormat format,
                                 VkExtent2D extent,
                                 VkImageUsageFlags usage,
                                 VkImageAspectFlags aspect,
                                 VkSampleCountFlagBits sample_count,
                                 double_ptr<vulkan::Image> pp_image) const {
  return CreateImage(format, extent, usage, aspect, sample_count, VMA_MEMORY_USAGE_GPU_ONLY, pp_image);
}

VkResult VulkanCore::CreateImage(VkFormat format,
                                 VkExtent2D extent,
                                 VkImageUsageFlags usage,
                                 VkImageAspectFlags aspect,
                                 double_ptr<vulkan::Image> pp_image) const {
  return CreateImage(format, extent, usage, aspect, VK_SAMPLE_COUNT_1_BIT, pp_image);
}

VkResult VulkanCore::CreateImage(VkFormat format,
                                 VkExtent2D extent,
                                 VkImageUsageFlags usage,
                                 double_ptr<vulkan::Image> pp_image) const {
  VkImageAspectFlagBits aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  if (IsDepthFormat(format)) {
    aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  return CreateImage(format, extent, usage, aspect, pp_image);
}

VkResult VulkanCore::CreateImage(VkFormat format, VkExtent2D extent, double_ptr<vulkan::Image> pp_image) const {
  VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  if (IsDepthFormat(format)) {
    usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  }

  return CreateImage(format, extent, usage, pp_image);
}

VkResult VulkanCore::CreateBuffer(VkDeviceSize size,
                                  VkBufferUsageFlags usage,
                                  VmaMemoryUsage memory_usage,
                                  VmaAllocationCreateFlags flags,
                                  VkDeviceSize alignment,
                                  double_ptr<vulkan::Buffer> pp_buffer) const {
  if (!pp_buffer) {
    SetErrorMessage("pp_buffer is nullptr");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = memory_usage;
  alloc_info.flags = flags;

  VkBuffer buffer;
  VmaAllocation allocation;

  if (alignment) {
    RETURN_IF_FAILED_VK(
        vmaCreateBufferWithAlignment(allocator_, &buffer_info, &alloc_info, alignment, &buffer, &allocation, nullptr),
        "failed to create buffer!");
  } else {
    RETURN_IF_FAILED_VK(vmaCreateBuffer(allocator_, &buffer_info, &alloc_info, &buffer, &allocation, nullptr),
                        "failed to create buffer!");
  }

  pp_buffer.construct(this, size, buffer, allocation);

  return VK_SUCCESS;
}

VkResult VulkanCore::CreateBuffer(VkDeviceSize size,
                                  VkBufferUsageFlags usage,
                                  VmaMemoryUsage memory_usage,
                                  VmaAllocationCreateFlags flags,
                                  double_ptr<vulkan::Buffer> pp_buffer) const {
  return CreateBuffer(size, usage, memory_usage, flags, 0, pp_buffer);
}

VkResult VulkanCore::CreateBuffer(VkDeviceSize size,
                                  VkBufferUsageFlags usage,
                                  VmaMemoryUsage memory_usage,
                                  double_ptr<vulkan::Buffer> pp_buffer) const {
  VmaAllocationCreateFlags flags = 0;
  return CreateBuffer(size, usage, memory_usage, flags, pp_buffer);
}

VkResult VulkanCore::CreatePipeline(const struct vulkan::PipelineSettings &settings, VkPipeline *pp_pipeline) const {
  if (!pp_pipeline) {
    SetErrorMessage("pp_pipeline is nullptr");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  VkPipelineVertexInputStateCreateInfo vertex_input_info{};
  vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input_info.vertexBindingDescriptionCount = settings.vertex_input_binding_descriptions.size();
  if (vertex_input_info.vertexBindingDescriptionCount) {
    vertex_input_info.pVertexBindingDescriptions = settings.vertex_input_binding_descriptions.data();
  }
  vertex_input_info.vertexAttributeDescriptionCount = settings.vertex_input_attribute_descriptions.size();
  if (vertex_input_info.vertexAttributeDescriptionCount) {
    vertex_input_info.pVertexAttributeDescriptions = settings.vertex_input_attribute_descriptions.data();
  }

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = nullptr;  // Dynamic
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = nullptr;  // Dynamic

  VkPipelineColorBlendStateCreateInfo color_blend_state{};
  color_blend_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blend_state.logicOpEnable = VK_FALSE;
  color_blend_state.logicOp = VK_LOGIC_OP_COPY;
  color_blend_state.attachmentCount = settings.pipeline_color_blend_attachment_states.size();
  color_blend_state.pAttachments = settings.pipeline_color_blend_attachment_states.data();
  color_blend_state.blendConstants[0] = 0.0f;
  color_blend_state.blendConstants[1] = 0.0f;
  color_blend_state.blendConstants[2] = 0.0f;
  color_blend_state.blendConstants[3] = 0.0f;

  std::vector<VkDynamicState> dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  if (settings.dynamic_primitive_topology) {
    dynamic_states.push_back(VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY);
  }

  VkPipelineDynamicStateCreateInfo dynamic_state{};
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
  if (dynamic_state.dynamicStateCount) {
    dynamic_state.pDynamicStates = dynamic_states.data();
  }

  VkGraphicsPipelineCreateInfo pipeline_create_info{};

  pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

  VkPipelineRenderingCreateInfoKHR rendering_create_info{};
  rendering_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
  rendering_create_info.colorAttachmentCount = settings.color_attachment_formats.size();
  rendering_create_info.pColorAttachmentFormats = settings.color_attachment_formats.data();
  rendering_create_info.depthAttachmentFormat = settings.depth_attachment_format;
  pipeline_create_info.pNext = &rendering_create_info;

  pipeline_create_info.stageCount = settings.shader_stage_create_infos.size();
  pipeline_create_info.pStages = settings.shader_stage_create_infos.data();
  pipeline_create_info.pVertexInputState = &vertex_input_info;
  pipeline_create_info.pInputAssemblyState = &settings.input_assembly_state_create_info;
  pipeline_create_info.pViewportState = &viewport_state;
  pipeline_create_info.pRasterizationState = &settings.rasterization_state_create_info;
  pipeline_create_info.pMultisampleState = &settings.multisample_state_create_info;
  pipeline_create_info.pColorBlendState = &color_blend_state;
  pipeline_create_info.pDynamicState = &dynamic_state;
  pipeline_create_info.layout = settings.pipeline_layout;
  pipeline_create_info.renderPass = VK_NULL_HANDLE;
  pipeline_create_info.pDepthStencilState = settings.depth_stencil_state_create_info.has_value()
                                                ? &settings.depth_stencil_state_create_info.value()
                                                : nullptr;
  pipeline_create_info.subpass = settings.subpass;
  pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
  pipeline_create_info.pTessellationState =
      settings.tessellation_state_create_info.has_value() ? &settings.tessellation_state_create_info.value() : nullptr;

  VkPipeline pipeline;
  RETURN_IF_FAILED_VK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &pipeline),
                      "failed to create graphics pipeline!");

  *pp_pipeline = pipeline;

  return VK_SUCCESS;
}

VkResult VulkanCore::CreateBottomLevelAccelerationStructure(VkDeviceAddress aabb_address,
                                                            VkDeviceSize stride,
                                                            uint32_t num_aabb,
                                                            VkGeometryFlagsKHR flags,
                                                            VkCommandPool command_pool,
                                                            VkQueue queue,
                                                            double_ptr<vulkan::AccelerationStructure> pp_blas) {
  const VkBufferUsageFlags buffer_usage_flags =
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  // Setup a single transformation matrix that can be used to transform the
  // whole geometry for a single bottom level acceleration structure
  VkTransformMatrixKHR transform_matrix = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  std::unique_ptr<vulkan::Buffer> transform_matrix_buffer;
  RETURN_IF_FAILED_VK(
      CreateBuffer(sizeof(transform_matrix), buffer_usage_flags, VMA_MEMORY_USAGE_CPU_TO_GPU, &transform_matrix_buffer),
      "failed to create transform matrix buffer!");
  std::memcpy(transform_matrix_buffer->Map(), &transform_matrix, sizeof(transform_matrix));
  transform_matrix_buffer->Unmap();

  VkDeviceOrHostAddressConstKHR aabb_data_device_address{};
  VkDeviceOrHostAddressConstKHR transform_matrix_device_address{};

  aabb_data_device_address.deviceAddress = aabb_address;
  transform_matrix_device_address.deviceAddress = transform_matrix_buffer->GetDeviceAddress();

  // The bottom level acceleration structure contains one set of triangles as
  // the input geometry
  VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
  acceleration_structure_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
  acceleration_structure_geometry.flags = flags;
  acceleration_structure_geometry.geometry.aabbs.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
  acceleration_structure_geometry.geometry.aabbs.pNext = nullptr;
  acceleration_structure_geometry.geometry.aabbs.stride = stride;
  acceleration_structure_geometry.geometry.aabbs.data = aabb_data_device_address;

  std::unique_ptr<vulkan::Buffer> buffer;
  VkAccelerationStructureKHR acceleration_structure;

  BuildAccelerationStructure(this, acceleration_structure_geometry, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                             VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                             VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR, num_aabb, command_pool, queue,
                             &acceleration_structure, &buffer);

  VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info{};
  acceleration_device_address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
  acceleration_device_address_info.accelerationStructure = acceleration_structure;
  VkDeviceAddress device_address =
      procedures_.vkGetAccelerationStructureDeviceAddressKHR(device_, &acceleration_device_address_info);
  pp_blas.construct(this, std::move(buffer), device_address, acceleration_structure, num_aabb);
  return VK_SUCCESS;
}

VkResult VulkanCore::CreateBottomLevelAccelerationStructure(VkDeviceAddress vertex_buffer_address,
                                                            VkDeviceAddress index_buffer_address,
                                                            uint32_t num_vertex,
                                                            VkDeviceSize stride,
                                                            uint32_t primitive_count,
                                                            VkGeometryFlagsKHR flags,
                                                            VkCommandPool command_pool,
                                                            VkQueue queue,
                                                            double_ptr<vulkan::AccelerationStructure> pp_blas) {
  const VkBufferUsageFlags buffer_usage_flags =
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  // Setup a single transformation matrix that can be used to transform the
  // whole geometry for a single bottom level acceleration structure
  VkTransformMatrixKHR transform_matrix = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  std::unique_ptr<vulkan::Buffer> transform_matrix_buffer;
  RETURN_IF_FAILED_VK(CreateBuffer(sizeof(transform_matrix), buffer_usage_flags, VMA_MEMORY_USAGE_CPU_TO_GPU, 0, 16,
                                   &transform_matrix_buffer),
                      "failed to create transform matrix buffer!");
  std::memcpy(transform_matrix_buffer->Map(), &transform_matrix, sizeof(transform_matrix));
  transform_matrix_buffer->Unmap();

  VkDeviceOrHostAddressConstKHR vertex_data_device_address{};
  VkDeviceOrHostAddressConstKHR index_data_device_address{};
  VkDeviceOrHostAddressConstKHR transform_matrix_device_address{};

  vertex_data_device_address.deviceAddress = vertex_buffer_address;
  index_data_device_address.deviceAddress = index_buffer_address;
  transform_matrix_device_address.deviceAddress = transform_matrix_buffer->GetDeviceAddress();

  // The bottom level acceleration structure contains one set of triangles as
  // the input geometry
  VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
  acceleration_structure_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  acceleration_structure_geometry.flags = flags;
  acceleration_structure_geometry.geometry.triangles.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  acceleration_structure_geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  acceleration_structure_geometry.geometry.triangles.vertexData = vertex_data_device_address;
  acceleration_structure_geometry.geometry.triangles.maxVertex = num_vertex;
  acceleration_structure_geometry.geometry.triangles.vertexStride = stride;
  acceleration_structure_geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
  acceleration_structure_geometry.geometry.triangles.indexData = index_data_device_address;
  acceleration_structure_geometry.geometry.triangles.transformData = transform_matrix_device_address;

  std::unique_ptr<vulkan::Buffer> buffer;
  VkAccelerationStructureKHR acceleration_structure;

  BuildAccelerationStructure(this, acceleration_structure_geometry, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                             VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                             VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR, primitive_count, command_pool, queue,
                             &acceleration_structure, &buffer);

  VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info{};
  acceleration_device_address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
  acceleration_device_address_info.accelerationStructure = acceleration_structure;
  VkDeviceAddress device_address =
      procedures_.vkGetAccelerationStructureDeviceAddressKHR(device_, &acceleration_device_address_info);
  pp_blas.construct(this, std::move(buffer), device_address, acceleration_structure, primitive_count);
  return VK_SUCCESS;
}

VkResult VulkanCore::CreateBottomLevelAccelerationStructure(VkDeviceAddress vertex_buffer_address,
                                                            VkDeviceAddress index_buffer_address,
                                                            uint32_t num_vertex,
                                                            VkDeviceSize stride,
                                                            uint32_t primitive_count,
                                                            VkCommandPool command_pool,
                                                            VkQueue queue,
                                                            double_ptr<vulkan::AccelerationStructure> pp_blas) {
  return CreateBottomLevelAccelerationStructure(vertex_buffer_address, index_buffer_address, num_vertex, stride,
                                                primitive_count, VK_GEOMETRY_OPAQUE_BIT_KHR, command_pool, queue,
                                                pp_blas);
}

VkResult VulkanCore::CreateBottomLevelAccelerationStructure(vulkan::Buffer *vertex_buffer,
                                                            vulkan::Buffer *index_buffer,
                                                            VkDeviceSize stride,
                                                            VkCommandPool command_pool,
                                                            VkQueue queue,
                                                            double_ptr<vulkan::AccelerationStructure> pp_blas) {
  return CreateBottomLevelAccelerationStructure(
      vertex_buffer->GetDeviceAddress(), index_buffer->GetDeviceAddress(), vertex_buffer->Size() / stride, stride,
      index_buffer->Size() / (sizeof(uint32_t) * 3), command_pool, queue, pp_blas);
}

VkResult VulkanCore::CreateTopLevelAccelerationStructure(
    const std::vector<VkAccelerationStructureInstanceKHR> &instances,
    VkCommandPool command_pool,
    VkQueue queue,
    double_ptr<vulkan::AccelerationStructure> pp_tlas) {
  std::unique_ptr<vulkan::Buffer> instances_buffer;
  CreateBuffer(
      sizeof(VkAccelerationStructureInstanceKHR) * std::max(instances.size(), static_cast<size_t>(1)),
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_CPU_TO_GPU, 0, 16, &instances_buffer);
  std::memcpy(instances_buffer->Map(), instances.data(), instances.size() * sizeof(VkAccelerationStructureInstanceKHR));
  instances_buffer->Unmap();

  VkDeviceOrHostAddressConstKHR instance_data_device_address{};
  instance_data_device_address.deviceAddress = instances_buffer->GetDeviceAddress();

  // The top level acceleration structure contains (bottom level) instance as
  // the input geometry
  VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
  acceleration_structure_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  acceleration_structure_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  acceleration_structure_geometry.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  acceleration_structure_geometry.geometry.instances.arrayOfPointers = VK_FALSE;
  acceleration_structure_geometry.geometry.instances.data = instance_data_device_address;

  VkAccelerationStructureKHR acceleration_structure;

  std::unique_ptr<vulkan::Buffer> buffer;

  BuildAccelerationStructure(
      this, acceleration_structure_geometry, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
      VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR, instances.size(), command_pool, queue, &acceleration_structure,
      &buffer);

  // Get the top acceleration structure's handle, which will be used to setup
  // it's descriptor
  VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info{};
  acceleration_device_address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
  acceleration_device_address_info.accelerationStructure = acceleration_structure;
  VkDeviceAddress device_address =
      procedures_.vkGetAccelerationStructureDeviceAddressKHR(device_, &acceleration_device_address_info);
  pp_tlas.construct(this, std::move(buffer), device_address, acceleration_structure, instances.size());
  return VK_SUCCESS;
}

VkResult VulkanCore::CreateTopLevelAccelerationStructure(
    const std::vector<std::pair<vulkan::AccelerationStructure *, glm::mat4>> &objects,
    VkCommandPool command_pool,
    VkQueue queue,
    double_ptr<vulkan::AccelerationStructure> pp_tlas) {
  std::vector<VkAccelerationStructureInstanceKHR> acceleration_structure_instances;
  acceleration_structure_instances.reserve(objects.size());
  for (int i = 0; i < objects.size(); i++) {
    auto &object = objects[i];
    VkAccelerationStructureInstanceKHR acceleration_structure_instance{};
    acceleration_structure_instance.transform = {object.second[0][0], object.second[1][0], object.second[2][0],
                                                 object.second[3][0], object.second[0][1], object.second[1][1],
                                                 object.second[2][1], object.second[3][1], object.second[0][2],
                                                 object.second[1][2], object.second[2][2], object.second[3][2]};
    acceleration_structure_instance.instanceCustomIndex = i;
    acceleration_structure_instance.mask = 0xFF;
    acceleration_structure_instance.instanceShaderBindingTableRecordOffset = 0;
    acceleration_structure_instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    acceleration_structure_instance.accelerationStructureReference = object.first->DeviceAddress();
    acceleration_structure_instances.push_back(acceleration_structure_instance);
  }

  return CreateTopLevelAccelerationStructure(acceleration_structure_instances, command_pool, queue, pp_tlas);
}

VkResult VulkanCore::CreateRayTracingPipeline(VkPipelineLayout pipeline_layout,
                                              VulkanShader *ray_gen_shader,
                                              const std::vector<VulkanShader *> &miss_shaders,
                                              const std::vector<vulkan::HitGroup> &hit_groups,
                                              const std::vector<VulkanShader *> &callable_shaders,
                                              VkPipeline *pp_pipeline) const {
  std::vector<VkPipelineShaderStageCreateInfo> shader_stage_create_infos;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
  // Ray generation group
  {
    VkRayTracingShaderGroupCreateInfoKHR ray_gen_group_ci{};
    ray_gen_group_ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    ray_gen_group_ci.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    ray_gen_group_ci.generalShader = shader_stage_create_infos.size();
    ray_gen_group_ci.closestHitShader = VK_SHADER_UNUSED_KHR;
    ray_gen_group_ci.anyHitShader = VK_SHADER_UNUSED_KHR;
    ray_gen_group_ci.intersectionShader = VK_SHADER_UNUSED_KHR;
    shader_groups.push_back(ray_gen_group_ci);
  }
  shader_stage_create_infos.push_back({
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      nullptr,
      0,
      VK_SHADER_STAGE_RAYGEN_BIT_KHR,
      ray_gen_shader->ModuleHandle(),
      ray_gen_shader->EntryPointRef().c_str(),
      nullptr,
  });

  for (auto miss_shader : miss_shaders) {
    // Ray miss group
    {
      VkRayTracingShaderGroupCreateInfoKHR miss_group_ci{};
      miss_group_ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      miss_group_ci.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      miss_group_ci.generalShader = shader_stage_create_infos.size();
      miss_group_ci.closestHitShader = VK_SHADER_UNUSED_KHR;
      miss_group_ci.anyHitShader = VK_SHADER_UNUSED_KHR;
      miss_group_ci.intersectionShader = VK_SHADER_UNUSED_KHR;
      shader_groups.push_back(miss_group_ci);
    }
    shader_stage_create_infos.push_back({
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_MISS_BIT_KHR,
        miss_shader->ModuleHandle(),
        miss_shader->EntryPointRef().c_str(),
        nullptr,
    });
  }

  for (auto hit_group : hit_groups) {
    // Ray closest hit group
    VkRayTracingShaderGroupCreateInfoKHR hit_group_ci{};
    hit_group_ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    hit_group_ci.type = hit_group.procedure ? VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
                                            : VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    hit_group_ci.generalShader = VK_SHADER_UNUSED_KHR;
    hit_group_ci.closestHitShader = VK_SHADER_UNUSED_KHR;
    hit_group_ci.anyHitShader = VK_SHADER_UNUSED_KHR;
    hit_group_ci.intersectionShader = VK_SHADER_UNUSED_KHR;

    hit_group_ci.closestHitShader = shader_stage_create_infos.size();
    shader_stage_create_infos.push_back({
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        hit_group.closest_hit_shader->ModuleHandle(),
        hit_group.closest_hit_shader->EntryPointRef().c_str(),
        nullptr,
    });

    if (hit_group.any_hit_shader) {
      hit_group_ci.anyHitShader = shader_stage_create_infos.size();
      shader_stage_create_infos.push_back({
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          nullptr,
          0,
          VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
          hit_group.any_hit_shader->ModuleHandle(),
          hit_group.any_hit_shader->EntryPointRef().c_str(),
          nullptr,
      });
    }

    if (hit_group.intersection_shader) {
      hit_group_ci.intersectionShader = shader_stage_create_infos.size();

      shader_stage_create_infos.push_back({
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          nullptr,
          0,
          VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
          hit_group.intersection_shader->ModuleHandle(),
          hit_group.intersection_shader->EntryPointRef().c_str(),
          nullptr,
      });
    }

    shader_groups.push_back(hit_group_ci);
  }

  for (auto callable_shader : callable_shaders) {
    // Ray miss group
    {
      VkRayTracingShaderGroupCreateInfoKHR callable_group_ci{};
      callable_group_ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      callable_group_ci.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      callable_group_ci.generalShader = shader_stage_create_infos.size();
      callable_group_ci.closestHitShader = VK_SHADER_UNUSED_KHR;
      callable_group_ci.anyHitShader = VK_SHADER_UNUSED_KHR;
      callable_group_ci.intersectionShader = VK_SHADER_UNUSED_KHR;
      shader_groups.push_back(callable_group_ci);
    }
    shader_stage_create_infos.push_back({
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_CALLABLE_BIT_KHR,
        callable_shader->ModuleHandle(),
        callable_shader->EntryPointRef().c_str(),
        nullptr,
    });
  }

  VkRayTracingPipelineCreateInfoKHR ray_tracing_pipeline_create_info{};
  ray_tracing_pipeline_create_info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
  ray_tracing_pipeline_create_info.stageCount = shader_stage_create_infos.size();
  ray_tracing_pipeline_create_info.pStages = shader_stage_create_infos.data();
  ray_tracing_pipeline_create_info.groupCount = shader_groups.size();
  ray_tracing_pipeline_create_info.pGroups = shader_groups.data();
  ray_tracing_pipeline_create_info.maxPipelineRayRecursionDepth = 1;
  ray_tracing_pipeline_create_info.layout = pipeline_layout;
  ray_tracing_pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;

  VkPipeline pipeline;
  RETURN_IF_FAILED_VK(procedures_.vkCreateRayTracingPipelinesKHR(device_, VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
                                                                 &ray_tracing_pipeline_create_info, nullptr, &pipeline),
                      "failed to create ray tracing pipeline!");

  *pp_pipeline = pipeline;

  return VK_SUCCESS;
}

VkResult VulkanCore::CreateShaderBindingTable(VkPipeline pipeline,
                                              size_t miss_shader_count,
                                              size_t hit_group_count,
                                              const std::vector<int32_t> &miss_shader_indices,
                                              const std::vector<int32_t> &hit_group_indices,
                                              const std::vector<int32_t> &callable_shader_indices,
                                              double_ptr<vulkan::ShaderBindingTable> pp_sbt) const {
  auto aligned_size = [](uint32_t value, uint32_t alignment) { return (value + alignment - 1) & ~(alignment - 1); };

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_pipeline_properties =
      physical_device_->GetPhysicalDeviceRayTracingPipelineProperties();

  const uint32_t handle_size = ray_tracing_pipeline_properties.shaderGroupHandleSize;
  const uint32_t handle_size_aligned = aligned_size(ray_tracing_pipeline_properties.shaderGroupHandleSize,
                                                    ray_tracing_pipeline_properties.shaderGroupHandleAlignment);
  const uint32_t base_alignment = ray_tracing_pipeline_properties.shaderGroupBaseAlignment;
  const uint32_t group_count =
      1 + miss_shader_indices.size() + hit_group_indices.size() + callable_shader_indices.size();
  VkDeviceSize raygen_shader_offset = 0;
  VkDeviceSize miss_shader_offset = 0;
  VkDeviceSize hit_group_offset = 0;
  VkDeviceSize callable_shader_offset = 0;
  VkDeviceSize sbt_size = 0;
  miss_shader_offset = sbt_size = aligned_size(sbt_size + handle_size_aligned * 1, base_alignment);
  hit_group_offset = sbt_size =
      aligned_size(sbt_size + handle_size_aligned * miss_shader_indices.size(), base_alignment);
  callable_shader_offset = sbt_size =
      aligned_size(sbt_size + handle_size_aligned * hit_group_indices.size(), base_alignment);
  sbt_size = aligned_size(sbt_size + handle_size_aligned * callable_shader_indices.size(), base_alignment);
  const VkBufferUsageFlags sbt_buffer_usage_flags = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  // Ray_gen
  // Create binding table buffers for each shader type
  std::unique_ptr<vulkan::Buffer> buffer;
  CreateBuffer(sbt_size, sbt_buffer_usage_flags, VMA_MEMORY_USAGE_CPU_TO_GPU, 0, base_alignment, &buffer);

  VkDeviceAddress buffer_address = buffer->GetDeviceAddress();

  // Copy the pipeline's shader handles into a host buffer
  std::vector<uint8_t> shader_handle_storage(handle_size * group_count);
  procedures_.vkGetRayTracingShaderGroupHandlesKHR(device_, pipeline, 0, group_count, handle_size * group_count,
                                                   shader_handle_storage.data());

  // Copy the shader handles from the host buffer to the binding tables
  auto *data = static_cast<uint8_t *>(buffer->Map());
  std::memcpy(data + raygen_shader_offset, shader_handle_storage.data(), handle_size_aligned);
  auto data_head = data + miss_shader_offset;
  for (auto miss_shader_index : miss_shader_indices) {
    std::memcpy(data_head, shader_handle_storage.data() + handle_size * (miss_shader_index + 1), handle_size);
    data_head += handle_size_aligned;
  }
  data_head = data + hit_group_offset;
  for (auto hit_group_index : hit_group_indices) {
    std::memcpy(data_head, shader_handle_storage.data() + handle_size * (hit_group_index + miss_shader_count + 1),
                handle_size);
    data_head += handle_size_aligned;
  }
  data_head = data + callable_shader_offset;
  for (auto callable_shader_index : callable_shader_indices) {
    std::memcpy(
        data_head,
        shader_handle_storage.data() + handle_size * (callable_shader_index + hit_group_count + miss_shader_count + 1),
        handle_size);
    data_head += handle_size_aligned;
  }
  buffer->Unmap();

  pp_sbt.construct(std::move(buffer), buffer_address + raygen_shader_offset, buffer_address + miss_shader_offset,
                   buffer_address + hit_group_offset, buffer_address + callable_shader_offset,
                   miss_shader_indices.size(), hit_group_indices.size(), callable_shader_indices.size());

  return VK_SUCCESS;
}

}  // namespace grassland::graphics::backend
