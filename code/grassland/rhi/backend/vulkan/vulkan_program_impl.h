#pragma once
#include "grassland/rhi/backend/backend_program_state.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/vulkan/vulkan_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base_state.h"

namespace grassland::rhi::backend::vulkan {
class VulkanProgramImpl final : public Object<IVulkanProgram> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IProgram, IVulkanProgram, IBackendProgram, IProgram, IVulkanProgramBase, IObject)
  static IVulkanProgram *Create(IVulkanDevice *device, const ProgramDesc &desc);
  ~VulkanProgramImpl() override;
  VkPipeline NativePipeline() const;

 private:
  VulkanProgramImpl(IVulkanDevice *device, const ProgramDesc &desc);
  VkPipeline pipeline_{};

 private:
  ProgramState state_Program_;

  ProgramState &StateProgram() override {
    return state_Program_;
  }

  const ProgramState &StateProgram() const override {
    return state_Program_;
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
  const ProgramDesc &GetDesc() const override {
    return IBackendProgram::CommonGetDesc();
  }
};
}  // namespace grassland::rhi::backend::vulkan
