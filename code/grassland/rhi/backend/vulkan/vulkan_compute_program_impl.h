#pragma once
#include "grassland/rhi/backend/backend_compute_program_state.h"
#include "grassland/rhi/backend/vulkan/vulkan_compute_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base_state.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanComputeProgramImpl final : public Object<IVulkanComputeProgram> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IComputeProgram,
                            IVulkanComputeProgram,
                            IBackendComputeProgram,
                            IComputeProgram,
                            IVulkanProgramBase,
                            IObject)
  static IVulkanComputeProgram *Create(IVulkanDevice *device, const ComputeProgramDesc &desc);
  ~VulkanComputeProgramImpl() override;
  VkPipeline NativePipeline() const;

 private:
  VulkanComputeProgramImpl(IVulkanDevice *device, const ComputeProgramDesc &desc);
  VkPipeline pipeline_{};

 private:
  ComputeProgramState state_ComputeProgram_;

  ComputeProgramState &StateComputeProgram() override {
    return state_ComputeProgram_;
  }

  const ComputeProgramState &StateComputeProgram() const override {
    return state_ComputeProgram_;
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
  const ComputeProgramDesc &GetDesc() const override {
    return IBackendComputeProgram::CommonGetDesc();
  }
};
}  // namespace grassland::rhi::backend::vulkan
