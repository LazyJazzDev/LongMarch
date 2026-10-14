#pragma once
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice final : public IDevice {
 public:
  static std::unique_ptr<VulkanDevice> CreateVulkanDevice(grassland::rhi::PhysicalDevice info,
                                                          ICore *core,
                                                          VkPhysicalDevice physical_device,
                                                          uint32_t api_version);
  ~VulkanDevice() override;
  VkDevice Device() const;
  VkPhysicalDevice PhysicalDevice() const;
  VkInstance Instance() const;
  VkQueue Queue() const;
  uint32_t QueueFamily() const;
  bool SupportsSwapchain() const;
  std::mutex &QueueMutex();
  std::unique_ptr<IWindow> CreateWindow(const WindowDesc &desc) override;
  std::unique_ptr<IQueue> CreateQueue(QueueType type) override;

 private:
  VulkanDevice(grassland::rhi::PhysicalDevice info, ICore *core);
  VkDevice device_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_{};
  VkQueue queue_{};
  uint32_t family_{};
  bool swapchain_{};
  std::mutex mutex_;
};
}  // namespace grassland::rhi::backend::vulkan
