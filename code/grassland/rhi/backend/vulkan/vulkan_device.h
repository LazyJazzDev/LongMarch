#pragma once
#include <tuple>

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_device_procedures.h"
#include "grassland/rhi/backend/vulkan/vulkan_render_target_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_scratch_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanDevice, "96a1c412-20e1-40bc-b542-d515f9a41965")

namespace grassland::rhi::backend::vulkan {
class IVulkanDevice : public IBackendDevice {
 public:
  static rhi::IDevice *CreateVulkanDevice(grassland::rhi::PhysicalDevice info,
                                          ICore *core,
                                          VkPhysicalDevice physical_device,
                                          uint32_t api_version);
  virtual VkDevice NativeDevice() const = 0;
  virtual VulkanRenderTargetPool &RenderTargets() = 0;
  virtual VulkanDescriptorPool &DescriptorPool() = 0;
  virtual const DeviceProcedures &Procedures() const = 0;
  virtual VmaAllocator Allocator() const = 0;
  virtual VkPhysicalDevice NativePhysicalDevice() const = 0;
  virtual VkInstance Instance() const = 0;
  virtual uint32_t QueueFamily() const = 0;
  virtual bool SupportsSwapchain() const = 0;
  virtual VulkanTemporalBufferPool &UploadPool() = 0;
  virtual VulkanTemporalBufferPool &DownloadPool() = 0;
  virtual VulkanScratchPool &ScratchPool() = 0;
  virtual  // Transfer idle native resources; callers recycle only after GPU completion and final handle release.
      // The pool grows on demand, independently of swapchain image count.
      bool
      AcquireCommandResources(VkCommandPool &pool, VkCommandBuffer &command) = 0;
  virtual void RecycleCommandResources(VkCommandPool pool, VkCommandBuffer command) = 0;
};

}  // namespace grassland::rhi::backend::vulkan
