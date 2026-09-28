#pragma once

#include "glm/glm.hpp"
#include "grassland/graphics/backend/vulkan/helper/device_creation_assist.h"
#include "grassland/graphics/backend/vulkan/helper/device_procedures.h"
#include "grassland/graphics/backend/vulkan/helper/instance.h"
#include "grassland/graphics/backend/vulkan/helper/physical_device.h"

namespace grassland::graphics::backend::vulkan {

class Device {
 public:
  Device(VkInstance instance,
         uint32_t api_version,
         InstanceProcedures instance_procedures,
         const class PhysicalDevice &physical_device,
         DeviceCreateInfo create_info,
         VmaAllocatorCreateFlags allocator_flags,
         VkDevice device);

  ~Device();

  VkDevice Handle() const {
    return device_;
  }

  VkInstance Instance() const {
    return instance_;
  }

  const class PhysicalDevice &PhysicalDevice() const {
    return physical_device_;
  }

  const DeviceCreateInfo &CreateInfo() const {
    return create_info_;
  }

  DeviceProcedures &Procedures() {
    return procedures_;
  }

  const DeviceProcedures &Procedures() const {
    return procedures_;
  }

  VmaAllocator Allocator() const {
    return allocator_;
  }

  VkResult WaitIdle() const {
    return vkDeviceWaitIdle(device_);
  }

  uint32_t SubGroupSize() const {
    return subgroup_properties_.subgroupSize;
  }

  VkResult CreateShaderModule(const CompiledShaderBlob &code, double_ptr<ShaderModule> pp_shader_module) const;

  VkResult CreateShaderModule(const void *p_code,
                              size_t code_size,
                              const std::string &entry_point,
                              double_ptr<ShaderModule> pp_shader_module) const;

  VkResult CreateDescriptorPool(const std::vector<VkDescriptorPoolSize> &pool_sizes,
                                uint32_t max_sets,
                                double_ptr<DescriptorPool> pp_descriptor_pool) const;

  VkResult CreateDescriptorSetLayout(const std::vector<VkDescriptorSetLayoutBinding> &bindings,
                                     double_ptr<DescriptorSetLayout> pp_descriptor_set_layout) const;

  VkResult CreateImage(VkFormat format,
                       VkExtent2D extent,
                       VkImageUsageFlags usage,
                       VkImageAspectFlags aspect,
                       VkSampleCountFlagBits sample_count,
                       VmaMemoryUsage mem_usage,
                       double_ptr<Image> pp_image) const;

  VkResult CreateImage(VkFormat format,
                       VkExtent2D extent,
                       VkImageUsageFlags usage,
                       VkImageAspectFlags aspect,
                       VkSampleCountFlagBits sample_count,
                       double_ptr<Image> pp_image) const;

  VkResult CreateImage(VkFormat format,
                       VkExtent2D extent,
                       VkImageUsageFlags usage,
                       VkImageAspectFlags aspect,
                       double_ptr<Image> pp_image) const;

  VkResult CreateImage(VkFormat format, VkExtent2D extent, VkImageUsageFlags usage, double_ptr<Image> pp_image) const;

  VkResult CreateImage(VkFormat format, VkExtent2D extent, double_ptr<Image> pp_image) const;

  VkResult CreateBuffer(VkDeviceSize size,
                        VkBufferUsageFlags usage,
                        VmaMemoryUsage memory_usage,
                        VmaAllocationCreateFlags flags,
                        VkDeviceSize alignment,
                        double_ptr<Buffer> pp_buffer) const;

  VkResult CreateBuffer(VkDeviceSize size,
                        VkBufferUsageFlags usage,
                        VmaMemoryUsage memory_usage,
                        VmaAllocationCreateFlags flags,
                        double_ptr<Buffer> pp_buffer) const;

  VkResult CreateBuffer(VkDeviceSize size,
                        VkBufferUsageFlags usage,
                        VmaMemoryUsage memory_usage,
                        double_ptr<Buffer> pp_buffer) const;

  VkResult CreatePipelineLayout(const std::vector<VkDescriptorSetLayout> &descriptor_set_layouts,
                                double_ptr<PipelineLayout> pp_pipeline_layout) const;

  VkResult CreatePipeline(const struct PipelineSettings &settings, double_ptr<Pipeline> pp_pipeline) const;

  VkResult CreateBottomLevelAccelerationStructure(VkDeviceAddress aabb_address,
                                                  VkDeviceSize stride,
                                                  uint32_t num_aabb,
                                                  VkGeometryFlagsKHR flags,
                                                  VkCommandPool command_pool,
                                                  VkQueue queue,
                                                  double_ptr<AccelerationStructure> pp_blas);

  VkResult CreateBottomLevelAccelerationStructure(VkDeviceAddress vertex_buffer_address,
                                                  VkDeviceAddress index_buffer_address,
                                                  uint32_t num_vertex,
                                                  VkDeviceSize stride,
                                                  uint32_t primitive_count,
                                                  VkGeometryFlagsKHR flags,
                                                  VkCommandPool command_pool,
                                                  VkQueue queue,
                                                  double_ptr<AccelerationStructure> pp_blas);

  VkResult CreateBottomLevelAccelerationStructure(VkDeviceAddress vertex_buffer_address,
                                                  VkDeviceAddress index_buffer_address,
                                                  uint32_t num_vertex,
                                                  VkDeviceSize stride,
                                                  uint32_t primitive_count,
                                                  VkCommandPool command_pool,
                                                  VkQueue queue,
                                                  double_ptr<AccelerationStructure> pp_blas);

  VkResult CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                                  Buffer *index_buffer,
                                                  VkDeviceSize stride,
                                                  VkCommandPool command_pool,
                                                  VkQueue queue,
                                                  double_ptr<AccelerationStructure> pp_blas);

  VkResult CreateTopLevelAccelerationStructure(const std::vector<VkAccelerationStructureInstanceKHR> &instances,
                                               VkCommandPool command_pool,
                                               VkQueue queue,
                                               double_ptr<AccelerationStructure> pp_tlas);

  VkResult CreateTopLevelAccelerationStructure(
      const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
      VkCommandPool command_pool,
      VkQueue queue,
      double_ptr<AccelerationStructure> pp_tlas);

  VkResult CreateRayTracingPipeline(PipelineLayout *pipeline_layout,
                                    ShaderModule *ray_gen_shader,
                                    const std::vector<ShaderModule *> &miss_shaders,
                                    const std::vector<HitGroup> &hit_groups,
                                    const std::vector<ShaderModule *> &callable_shaders,
                                    double_ptr<RayTracingPipeline> pp_pipeline) const;

  VkResult CreateRayTracingPipeline(PipelineLayout *pipeline_layout,
                                    ShaderModule *ray_gen_shader,
                                    ShaderModule *miss_shader,
                                    ShaderModule *closest_hit_shader,
                                    double_ptr<RayTracingPipeline> pp_pipeline) const;

  VkResult CreateShaderBindingTable(RayTracingPipeline *ray_tracing_pipeline,
                                    const std::vector<int32_t> &miss_shader_indices,
                                    const std::vector<int32_t> &hit_group_indices,
                                    const std::vector<int32_t> &callable_shader_indices,
                                    double_ptr<ShaderBindingTable> pp_sbt) const;

  VkResult CreateShaderBindingTable(RayTracingPipeline *ray_tracing_pipeline,
                                    double_ptr<ShaderBindingTable> pp_sbt) const;

  void NameObject(VkImage image, const std::string &name);
  void NameObject(VkImageView image_view, const std::string &name);
  void NameObject(VkBuffer buffer, const std::string &name);
  void NameObject(VkDeviceMemory memory, const std::string &name);
  void NameObject(VkPipeline pipeline, const std::string &name);
  void NameObject(VkPipelineLayout pipeline_layout, const std::string &name);
  void NameObject(VkDescriptorSetLayout descriptor_set_layout, const std::string &name);
  void NameObject(VkDescriptorSet descriptor_set, const std::string &name);
  void NameObject(VkRenderPass render_pass, const std::string &name);
  void NameObject(VkSampler sampler, const std::string &name);
  void NameObject(VkCommandPool command_pool, const std::string &name);
  void NameObject(VkCommandBuffer command_buffer, const std::string &name);
  void NameObject(VkFramebuffer framebuffer, const std::string &name);
  void NameObject(VkDescriptorPool descriptor_pool, const std::string &name);
  void NameObject(VkShaderModule shader_module, const std::string &name);
  void NameObject(VkAccelerationStructureKHR acceleration_structure, const std::string &name);

 private:
  VkInstance instance_{};
  uint32_t api_version_{};
  InstanceProcedures instance_procedures_{};

  class PhysicalDevice physical_device_;

  const DeviceCreateInfo create_info_;

  VkPhysicalDeviceSubgroupProperties subgroup_properties_{};

  VkDevice device_{};

  DeviceProcedures procedures_{};

  VmaAllocator allocator_{};
};

}  // namespace grassland::graphics::backend::vulkan
