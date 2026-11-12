#pragma once
#include <array>

#include "grassland/rhi/backend/backend_shader_binding_table.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanRayTracingProgram,
                        grassland::rhi::backend::vulkan::IVulkanProgramBase,
                        "dbbe212c-3234-418b-b6b0-6676da637f09")

namespace grassland::rhi::backend::vulkan {
class IVulkanRayTracingProgram : public IVulkanProgramBase {
 public:
  static rhi::IRayTracingProgram *Create(IVulkanDevice *device, const RayTracingProgramDesc &desc);
  virtual const IBackendShaderBindingTable *DefaultShaderBindingTable() const = 0;
  virtual VkPipeline NativePipeline() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan

namespace grassland::rhi::backend::vulkan {
class VulkanRayTracingProgramImpl final : public Object<rhi::IRayTracingProgram, IVulkanRayTracingProgram> {
 public:
  static rhi::IRayTracingProgram *Create(IVulkanDevice *device, const RayTracingProgramDesc &desc);

  IShaderBindingTable *CreateShaderBindingTable(const ShaderBindingTableDesc &desc) override;

  const IBackendShaderBindingTable *DefaultShaderBindingTable() const;

  ~VulkanRayTracingProgramImpl() override;

  VkPipeline NativePipeline() const;

  const RayTracingProgramDesc &GetDesc() const override;

  IVulkanDevice *Owner() const override;

  VkPipelineLayout NativeLayout() const override;

 private:
  VulkanRayTracingProgramImpl(IVulkanDevice *device, const RayTracingProgramDesc &desc);

  VkPipeline pipeline_{};

  IShaderBindingTable *BuildShaderBindingTable(const ShaderBindingTableDesc &desc);

  ObjectPtr<IBackendShaderBindingTable> table_;

  std::array<std::vector<uint8_t>, 4> identifiers_;

  uint64_t handle_size_{}, stride_{}, alignment_{};

  void InitializeRayTracingProgram(IVulkanDevice *device, const RayTracingProgramDesc &desc);

  bool ValidateLayout() const;

  bool ValidateShaderBindingTable(const ShaderBindingTableDesc &desc) const;

  ShaderBindingTableDesc DefaultShaderBindingTableDesc() const;

  IVulkanDevice *device_{};

  RayTracingProgramDesc desc_{};

  std::vector<ObjectPtr<IShader>> shaders_{};

  void InitializeVulkanProgramBase(VkDevice device);

  void DestroyVulkanProgramBase();

  bool CreateLayout(const PipelineLayoutDesc &desc);

  VkDevice LayoutDevice() const;

  VkDevice layout_device_{};
  std::vector<VkDescriptorSetLayout> set_layouts_{};
  VkPipelineLayout layout_{};
};
}  // namespace grassland::rhi::backend::vulkan
