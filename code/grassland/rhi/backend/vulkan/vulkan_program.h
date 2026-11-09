#pragma once
#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanProgram final : public Program, public VulkanProgramBase {
 public:
  void AddRef() noexcept override {
    RetainReference();
  }

  void Release() noexcept override {
    if (ReleaseReference())
      delete this;
  }

  const ProgramDesc &GetDesc() const override {
    return Description();
  }

  static VulkanProgram *Create(VulkanDevice *device, const ProgramDesc &desc);
  ~VulkanProgram() override;
  VkPipeline NativePipeline() const;

 private:
  VulkanProgram(VulkanDevice *device, const ProgramDesc &desc);
  VkPipeline pipeline_{};
};
}  // namespace grassland::rhi::backend::vulkan
