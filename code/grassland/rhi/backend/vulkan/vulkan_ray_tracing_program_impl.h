#pragma once
#include "grassland/rhi/backend/backend_ray_tracing_program_state.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base_state.h"
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

 private:
  VulkanRayTracingProgramImpl(IVulkanDevice *device, const RayTracingProgramDesc &desc);
  VkPipeline pipeline_{};
  IBackendShaderBindingTable *BuildShaderBindingTable(const ShaderBindingTableDesc &desc);
  ObjectPtr<IBackendShaderBindingTable> table_;
  std::array<std::vector<uint8_t>, 4> identifiers_;
  uint64_t handle_size_{}, stride_{}, alignment_{};

 private:
  RayTracingProgramState state_RayTracingProgram_;

  RayTracingProgramState &StateRayTracingProgram() override {
    return state_RayTracingProgram_;
  }

  const RayTracingProgramState &StateRayTracingProgram() const override {
    return state_RayTracingProgram_;
  }

 private:
  VulkanProgramBaseState state_VulkanProgramBase_;

  VulkanProgramBaseState &StateVulkanProgramBase() override {
    return state_VulkanProgramBase_;
  }

  const VulkanProgramBaseState &StateVulkanProgramBase() const override {
    return state_VulkanProgramBase_;
  }

 public:
  const RayTracingProgramDesc &GetDesc() const override {
    return IBackendRayTracingProgram::CommonGetDesc();
  }
};
}  // namespace grassland::rhi::backend::vulkan
