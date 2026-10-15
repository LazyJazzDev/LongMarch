#pragma once
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice final : public Device {
 public:
  static std::unique_ptr<VulkanDevice> CreateVulkanDevice(grassland::rhi::PhysicalDevice info,
                                                          Core *core,
                                                          VkPhysicalDevice physical_device,
                                                          uint32_t api_version);
  ~VulkanDevice() override;
  VkDevice NativeDevice() const;
  VkPhysicalDevice NativePhysicalDevice() const;
  VkInstance Instance() const;
  VkQueue NativeQueue() const;
  uint32_t QueueFamily() const;
  bool SupportsSwapchain() const;
  std::mutex &QueueMutex();
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  rhi::ICommandContext *CreateCommandContext() override;
  rhi::IQueue *GetQueue(QueueType type) override;

 private:
  VulkanDevice(grassland::rhi::PhysicalDevice info, Core *core);
  VkDevice device_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_{};
  VkQueue queue_{};
  uint32_t family_{};
  bool swapchain_{};
  std::mutex mutex_;
};
}  // namespace grassland::rhi::backend::vulkan
