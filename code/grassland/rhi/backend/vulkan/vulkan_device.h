#pragma once
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

class VulkanDevice final : public IDevice {
 public:
  class BackendLifetime {
   public:
    BackendLifetime(VkDevice device,
                    VkPhysicalDevice physical,
                    const ObjectPtr<ICore> &core,
                    uint32_t family,
                    bool swapchain);
    ~BackendLifetime();
    VkDevice Device() const;
    VkPhysicalDevice PhysicalDevice() const;
    VkInstance Instance() const;
    VkQueue Queue() const;
    uint32_t QueueFamily() const;
    bool SupportsSwapchain() const;
    std::mutex &QueueMutex();

   private:
    ObjectPtr<ICore> core_;
    VkDevice device_;
    VkPhysicalDevice physical_;
    VkQueue queue_{};
    uint32_t family_;
    bool swapchain_;
    std::mutex mutex_;
  };

  static std::unique_ptr<VulkanDevice> CreateVulkanDevice(PhysicalDevice info,
                                                          const ObjectPtr<ICore> &core,
                                                          VkPhysicalDevice physical_device,
                                                          uint32_t api_version);
  ~VulkanDevice() override;
  const std::shared_ptr<BackendLifetime> &NativeLifetime() const;
  const void *NativeIdentity() const override;
  std::unique_ptr<IWindow> CreateWindow(const WindowDesc &desc) override;
  std::unique_ptr<IQueue> CreateQueue(QueueType type) override;

 private:
  VulkanDevice(PhysicalDevice info, const ObjectPtr<ICore> &core);
  VkDevice device_{VK_NULL_HANDLE};
  std::shared_ptr<BackendLifetime> lifetime_;
};

}  // namespace grassland::rhi::backend::vulkan
