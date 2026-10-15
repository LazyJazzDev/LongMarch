#pragma once
#include <tuple>

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

  // Transfer idle native resources; callers recycle only after GPU completion and final handle release.
  // The pool grows on demand, independently of swapchain image count.
  bool AcquireCommandResources(VkCommandPool &pool, VkCommandBuffer &command, VkFence &fence);
  void RecycleCommandResources(VkCommandPool pool, VkCommandBuffer command, VkFence fence);

 private:
  std::mutex resource_pool_mutex_;
  std::vector<std::tuple<VkCommandPool, VkCommandBuffer, VkFence>> command_pool_;
  VulkanDevice(grassland::rhi::PhysicalDevice info, Core *core);
  VkDevice device_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_{};
  VkQueue queue_{};
  uint32_t family_{};
  bool swapchain_{};
  std::mutex mutex_;
};
}  // namespace grassland::rhi::backend::vulkan
