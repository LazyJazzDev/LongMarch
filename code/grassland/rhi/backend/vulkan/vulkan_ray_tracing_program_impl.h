#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_ray_tracing_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanRayTracingProgramImpl final : public Object<IVulkanRayTracingProgram> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IRayTracingProgram,
                            IVulkanRayTracingProgram,
                            IBackendRayTracingProgram,
                            IRayTracingProgram,
                            IVulkanProgramBase,
                            IObject)

  static IVulkanRayTracingProgram *Create(IVulkanDevice *device, const RayTracingProgramDesc &desc);

  IShaderBindingTable *CreateShaderBindingTable(const ShaderBindingTableDesc &desc) override;

  const IBackendShaderBindingTable *DefaultShaderBindingTable() const;

  ~VulkanRayTracingProgramImpl() override;

  VkPipeline NativePipeline() const;

  const RayTracingProgramDesc &GetDesc() const override;

  IBackendDevice *Owner() const override;

  VkPipelineLayout NativeLayout() const override;

 private:
  VulkanRayTracingProgramImpl(IVulkanDevice *device, const RayTracingProgramDesc &desc);

  VkPipeline pipeline_{};

  IBackendShaderBindingTable *BuildShaderBindingTable(const ShaderBindingTableDesc &desc);

  ObjectPtr<IBackendShaderBindingTable> table_;

  std::array<std::vector<uint8_t>, 4> identifiers_;

  uint64_t handle_size_{}, stride_{}, alignment_{};

  void InitializeRayTracingProgram(IBackendDevice *device, const RayTracingProgramDesc &desc);

  bool ValidateLayout() const;

  bool ValidateShaderBindingTable(const ShaderBindingTableDesc &desc) const;

  ShaderBindingTableDesc DefaultShaderBindingTableDesc() const;

  IBackendDevice *device_{};

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
